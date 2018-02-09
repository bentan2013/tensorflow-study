import tensorflow as tf


state = tf.Variable(0, name="connter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

for i in range(10):
    sess.run(update)
    print(sess.run(state))
