

#demo_for_attacks

def step_fgsm(x, eps, logits):
  label = tf.argmax(logits,1)
  one_hot_label = tf.one_hot(label, NUM_CLASSES)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot_label,
                                                  logits,
                                                  label_smoothing=0.1,
                                                  weights=1.0)
  x_adv = x + eps*tf.sign(tf.gradients(cross_entropy,x)[0])
  x_adv = tf.clip_by_value(x_adv,-1.0,1.0)
  return tf.stop_gradient(x_adv)


#BIM
def step_targeted_attack(x, eps, one_hot_target_class, logits):
  #one_hot_target_class = tf.one_hot(target, NUM_CLASSES)
  #print(one_hot_target_class,"\n\n")
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                  logits,
                                                  label_smoothing=0.1,
                                                  weights=1.0)
  x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
  x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
  return tf.stop_gradient(x_adv)


#Iterative Least-Likely Class Method

def step_ll_adversarial_images(x, eps, logits):
  least_likely_class = tf.argmin(logits, 1)
  one_hot_ll_class = tf.one_hot(least_likely_class, NUM_CLASSES)
  one_hot_ll_class = tf.reshape(one_hot_ll_class,[1,NUM_CLASSES])
  # This reuses the method described above
  return step_targeted_attack(x, eps, one_hot_ll_class,logits)



 
def run_inference_on_image(image):
  """Runs inference on an image.
 
  Args:
    image: Image file name.
 
  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()
  original_shape = cv2.imread(image).shape
  # Creates graph from saved GraphDef.
  create_graph()
  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    image_tensor = sess.graph.get_tensor_by_name('Mul:0')
    image = sess.run(image_tensor,{'DecodeJpeg/contents:0': image_data})
    predictions = sess.run(softmax_tensor,
                           {'Mul:0': image})
    predictions = np.squeeze(predictions)
    print("Generating Adversial Example...\n\n")
    target_class = tf.reshape(tf.one_hot(972,NUM_CLASSES),[1,NUM_CLASSES])
    adv_image_tensor,noise = step_targeted_attack(image_tensor, 0.007, target_class, softmax_tensor)
    #adv_image_tensor,noise = step_ll_adversarial_images(image_tensor, 0.007, softmax_tensor)
    #adv_image_tensor,noise = step_fgsm(image_tensor, 0.007, softmax_tensor)
    #adv_image = sess.run(adv_image_tensor,{'DecodeJpeg/contents:0': image_data})
    adv_image = image
    adv_noise = np.zeros(image.shape)
    for i in range(10):
        print("Iteration "+str(i))
        adv_image,a = sess.run((adv_image_tensor,noise),{'Mul:0': adv_image})
        adv_noise = adv_noise + a
 
    plt.imshow(image[0]/2 + 0.5)
    #plt.show()
    save_image(image,original_shape,"original.jpg")
    plt.imshow(adv_image[0]/2 + 0.5)
    #plt.show()
    save_image(adv_image,original_shape,"adv_image.jpg")
    plt.imshow(adv_noise[0]/2 + 0.5)
    #plt.show()
    save_image(adv_noise,original_shape,"adv_noise.jpg")
    
    adv_predictions = sess.run(softmax_tensor, {'Mul:0' : adv_image})
    adv_predictions = np.squeeze(adv_predictions)
    
    noise_predictions = sess.run(softmax_tensor, {'Mul:0' : adv_noise})
    noise_predictions = np.squeeze(noise_predictions)
    
    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()
 
    print("\nNormal Image ...\n")
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        score = predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))
 
    print("\nAdversial Image ...\n")
    top_k = adv_predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        score = adv_predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))
 
    print("\nAdversial Noise ...\n")
    top_k = noise_predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        score = noise_predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))