from matplotlib import pyplot as plt

all_loss_histories = [-0.476104647850886, -0.5163953955998953, -0.5823875085100336, -0.5856097577308156, -0.5747788693638014, 
                      -0.5737551336266279, -0.5889408824491154, -0.6423607066360082, -0.6107989444556514, -0.6470236311413643]
plt.plot([f"Gen {i}" for i in range(1, 11)], all_loss_histories, label="Winning Structure")
plt.title("Loss History for Each Generation Winning Structure")
plt.xlabel("Evolutionary Generation")
plt.ylabel("Loss")
plt.legend()
plt.show()