import RL

print "Test Solve"
large = RL.LargeRL(learning_rate = 0.1, max_iterations=100000, epochs=5)

large.solve()
print "Test Solve Complete"
