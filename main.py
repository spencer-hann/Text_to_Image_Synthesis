import gan

#exp1 doubled lrs
print("Starting experiment 1 higher learning rate")
print("*"*75)
lr = 0.0006
fname = 'exp1'
subtitle = "Learning Rate 0.002"
gan.train_model(lr, lr, fname, subtitle)

#exp 2 higher Generator learning rate
print("Starting experiment 2 higher generator learning rate")
print("*"*75)
lrG = 0.0006
lrD = 0.0002
fname = 'exp2'
subtitle = "Higher Generator Learning Rate"
gan.train_model(lrD, lrG, fname, subtitle)

#exp 3 higher Discriminator learning rate
print("Starting experiment 2 higher discriminator learning rate")
print("*"*75)
lrG = 0.0002
lrD = 0.0006
fname = 'exp3'
subtitle = "Higher Discriminator Learning Rate"
gan.train_model(lrD, lrG, fname, subtitle)
