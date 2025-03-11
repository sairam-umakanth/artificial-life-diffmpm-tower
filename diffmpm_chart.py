from matplotlib import pyplot as plt

loss_history_1 = [

all_loss_histories = [loss_history_1, loss_history_2]
for idx, history in enumerate(all_loss_histories):
        plt.plot(history, label=f'Structure {idx}')
plt.title("Loss History for Each Generation 2 Structure")
plt.xlabel("Gradient Descent Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()