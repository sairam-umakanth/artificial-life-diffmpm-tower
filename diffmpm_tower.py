import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import pickle

CHECKPOINT_FILE = "diffmpm_checkpoint.pkl"

real = ti.f64
ti.init(default_fp=real, arch=ti.cpu, flatten_if=True)

dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8
target = [0.8, 0.8]

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

n_sin_waves = 4
weights = scalar()
bias = scalar()
x_avg = vec()

actuation = scalar()
actuation_omega = 20
act_strength = 4

def create_video_from_images(image_folder, output_video, fps=20):
    # List and sort all .png images in the folder.
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    
    if not images:
        print("No images found in", image_folder)
        return

    # Read the first image to get the video dimensions.
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Define the codec and create the VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()
    print("Video saved as", output_video)

def save_checkpoint(gen, best_loss, best_structure_params, all_loss_histories, best_params_per_gen):
    checkpoint_data = {
        "gen": gen,
        "best_loss": best_loss,
        "best_structure_params": best_structure_params,
        "all_loss_histories": all_loss_histories,
        "best_params_per_gen": best_params_per_gen
    }
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved at generation {gen}.")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            checkpoint_data = pickle.load(f)
        print(f"Resuming from checkpoint at generation {checkpoint_data['gen']}.")
        return checkpoint_data
    return None

def reset_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint reset.")
    else:
        print("No checkpoint to reset.")

def allocate_fields():
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)

    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)

    ti.root.lazy_grad()

def reinitialize_taichi():
    ti.reset()
    ti.init(default_fp=real, arch=ti.cpu, flatten_if=True)
    global weights, bias, actuator_id, particle_type, x, v, grid_v_in, grid_m_in, grid_v_out, C, F, loss, x_avg, actuation
    # Re-declare all global fields:
    weights = ti.field(real)
    bias = ti.field(real)
    actuator_id = ti.field(ti.i32)
    particle_type = ti.field(ti.i32)
    x = vec()
    v = vec()
    grid_v_in = vec()
    grid_m_in = scalar()
    grid_v_out = vec()
    C = mat()
    F = mat()
    loss = scalar()
    x_avg = vec()
    actuation = scalar()
    allocate_fields()


@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]

        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        # ti.print(act)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base +
                          offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 3
coeff = 0.5


@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            act += weights[i, j] * ti.sin(actuation_omega * t * dt +
                                          2 * math.pi / n_sin_waves * j)
        act += bias[i]
        actuation[t, i] = ti.tanh(act)


@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])

@ti.kernel
def compute_loss():
    X_dist = x_avg[None][0]
    Y_dist = x_avg[None][1] 
    loss[None] = -X_dist - Y_dist

@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)


def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    compute_x_avg()
    compute_loss()


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def modify_material_properties(self, x, y, w, h, shape="rectangle", actuator_id=1):
        """
        Modifies the material properties of a sub-region within an already created body.

        Args:
            scene (Scene): The scene object containing particles.
            x (float): X-coordinate of the sub-region center.
            y (float): Y-coordinate of the sub-region center.
            w (float): Width of the sub-region (or diameter if circular).
            h (float): Height of the sub-region (only for rectangles).
            shape (str): Shape of the sub-region ("rectangle" or "circle").
            new_material_id (int): New material ID to assign.
        """
        for i in range(self.n_particles):
            px, py = self.x[i]  # Get particle position

            if shape == "rectangle":
                if (x - w / 2) <= px <= (x + w / 2) and (y - h / 2) <= py <= (y + h / 2):
                    self.actuator_id[i] = actuator_id
            elif shape == "circle":
                if (px - x) ** 2 + (py - y) ** 2 <= (w / 2) ** 2:
                    self.actuator_id[i] = actuator_id
            else:
                raise ValueError("Invalid shape! Use 'rectangle' or 'circle'.")


    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act


def fish(scene):
    scene.add_rect(0.025, 0.025, 0.95, 0.1, -1, ptype=0)
    scene.add_rect(0.1, 0.2, 0.15, 0.05, -1)
    scene.add_rect(0.1, 0.15, 0.025, 0.05, 0)
    scene.add_rect(0.125, 0.15, 0.025, 0.05, 1)
    scene.add_rect(0.2, 0.15, 0.025, 0.05, 2)
    scene.add_rect(0.225, 0.15, 0.025, 0.05, 3)
    scene.set_n_actuators(4)


def robot(scene, legs, layers):
    scene.set_offset(0.1, 0.03)
    
    body_width = 0.3
    body_height = 0.1
    leg_width = 0.05
    leg_height = 0.1
    spacing = (body_width - (legs * leg_width)) / (legs - 1) if legs > 1 else 0

    # Create the first set of legs (ground level)
    for i in range(legs):
        x_offset = i * (leg_width + spacing)
        scene.add_rect(x_offset, 0.0, leg_width, leg_height, i)

    # Start building layers
    for layer in range(layers):
        y_offset = (layer * (leg_height + body_height)) + leg_height  # Adjusted to be above the previous layer's legs
        
        # Place the main body at the correct height
        scene.add_rect(0.0, y_offset, body_width, body_height, -1)
        
        # Add leg-like structures above the current layer (except for the last one)
        if layer < layers - 1:
            for i in range(legs):
                x_offset = i * (leg_width + spacing)
                scene.add_rect(x_offset, y_offset + body_height, leg_width, leg_height, i)

    scene.set_n_actuators(legs*2)


gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)


def visualize(s, folder):
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=20, help="Inner gradient descent iterations")
    parser.add_argument('--gen_iters', type=int, default=10, help="Number of structures in generation")
    options = parser.parse_args()

    total_structures = options.gen_iters
    best_loss = float('inf')
    best_structure_params = None
    all_loss_histories = []
    best_params_per_gen = []

    checkpoint = load_checkpoint()

    if checkpoint:
        gen_start = checkpoint["gen"] + 1  # Resume from the next generation
        best_loss = checkpoint["best_loss"]
        best_structure_params = checkpoint["best_structure_params"]
        all_loss_histories = checkpoint["all_loss_histories"]
        best_params_per_gen = checkpoint["best_params_per_gen"]
    else:
        gen_start = 0  # Start from scratch
        best_loss = float('inf')
        best_structure_params = None
        all_loss_histories = []
        best_params_per_gen = []

    for gen in range(gen_start, total_structures):
        print(f"\n=== Generation Structure {gen} ===")
        # Create and configure a new scene.
        scene = Scene()
        legs =  5 #np.random.randint(2, 7)
        layers = 1 #np.random.randint(1, 4)
        robot(scene, legs, layers)
        # Add random modifications.
        discrepencies = np.random.randint(1, 5)
        for i in range(discrepencies):
            x1 = 0.1 + np.random.rand() * 0.3
            x2 = 0.1 + np.random.rand() * 0.3
            y1 = 0.03 + np.random.rand() * 0.2 * layers
            y2 = 0.03 + np.random.rand() * 0.2 * layers
            w1 = np.random.rand() * 0.1
            w2 = np.random.rand() * 0.2
            scene.modify_material_properties(x1, y1, w1, w1, "circle", -1)
            scene.modify_material_properties(x2, y2, w2, w2, "circle", 3)
        scene.finalize()
        reinitialize_taichi()  # Reinitialize Taichi fields.

        # Initialize actuator parameters.
        for i in range(n_actuators):
            for j in range(n_sin_waves):
                weights[i, j] = np.random.randn() * 0.01
            bias[i] = np.random.randn() * 0.01

        # Initialize particle state.
        for i in range(scene.n_particles):
            x[0, i] = scene.x[i]
            F[0, i] = [[1, 0], [0, 1]]
            actuator_id[i] = scene.actuator_id[i]
            particle_type[i] = scene.particle_type[i]

        # Inner gradient descent loop.
        structure_loss_history = []
        for iter in range(options.iters):
            with ti.ad.Tape(loss):
                forward()
            l = loss[None]
            structure_loss_history.append(l)
            print(f"Structure {gen}, iter {iter}: loss = {l}")

            learning_rate = 0.3
            for i in range(n_actuators):
                for j in range(n_sin_waves):
                    weights[i, j] -= learning_rate * weights.grad[i, j]
                bias[i] -= learning_rate * bias.grad[i]

            # Visualize at first and last iteration.
            if iter == 0 or iter == options.iters - 1:
                forward(1500)
                folder_name = f'diffmpm/iter{iter:03d}/'
                for s in range(15, 1500, 16):
                    visualize(s, folder_name)

        all_loss_histories.append(structure_loss_history)
        final_loss = structure_loss_history[-1]

        current_best = {
            'generation_iteration': gen,
            'legs': legs,
            'discrepancies': discrepencies,
            'layers': layers,
            'weights': weights.to_numpy().tolist(),
            'bias': bias.to_numpy().tolist(),
            'best_loss': final_loss
        }
        best_params_per_gen.append(current_best)
        if final_loss < best_loss:
            best_loss = final_loss
            best_structure_params = current_best

        # Record video for this structure(using one of the visualization folders)
        # Here, we're assuming you saved images in folder 'diffmpm/iter000/' for the first iteration.
        create_video_from_images('diffmpm/iter000/', f'evol6_structure_gen{gen:03d}_iter0.mp4', fps=20)
        create_video_from_images('diffmpm/iter019/', f'evol6_structure_gen{gen:03d}_iter20.mp4', fps=20)

        ti.reset()

        save_checkpoint(gen, best_loss, best_structure_params, all_loss_histories, best_params_per_gen)

    # Save best parameters per generation.
    with open('best_params.json', 'w') as f:
        json.dump(best_params_per_gen, f, indent=4)
    
    print("\nBest Performing Structure:")
    print(best_structure_params)


    # Plot loss histories.
    for idx, history in enumerate(all_loss_histories):
        plt.plot(history, label=f'Structure {idx}')
    plt.title("Loss History for Each Generation 6 Structure")
    plt.xlabel("Gradient Descent Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    print("\nBest Performing Structure:")
    print(best_structure_params)

    reset_checkpoint()

if __name__ == '__main__':
    main()