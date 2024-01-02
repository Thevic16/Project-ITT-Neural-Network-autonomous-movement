# Project-ITT-Neural-Network-autonomous-movement

[GitHub - Thevic16/Project-ITT-Neural-Network-autonomous-movement](https://github.com/Thevic16/Project-ITT-Neural-Network-autonomous-movement)

**Description:**

The Autonomous Intelligent Wheelchair with Predefined Route Navigation, implemented in Python, is designed to provide users with a seamless and reliable mobility solution. This project introduces a groundbreaking approach to autonomous movements, focusing on predefined routes in a known environment. Leveraging the power of neural networks for image processing, the wheelchair navigates through familiar spaces with precision and adaptability.

**Key Features:**

1. **Predefined Route Navigation:**
    - Allows users to define specific routes within a known environment, enhancing predictability and control.
2. **Autonomous Movements:**
    - Employs advanced neural networks for real-time image processing to interpret the environment.
    - Navigates the wheelchair autonomously along predefined routes, adapting to obstacles and changes in the environment.
3. **Neural Network Integration:**
    - **Data Collection Step:** Gathers datasets specific to the known environment, capturing diverse scenarios and potential obstacles.
    - **Training Step:** Utilizes machine learning techniques to train the neural network for accurate image processing and decision-making within the specified environment.
    - **Implementation Step:** Integrates the trained neural network into the wheelchair's control system for real-time route navigation.

**Why Neural Networks:**

- Neural networks enhance the wheelchair's ability to interpret and respond to the surrounding environment.
- Enables the wheelchair to make dynamic decisions based on real-time visual data, ensuring efficient navigation through predefined routes.
- Improves adaptability to changes in the environment, enhancing user safety and convenience.

**Skills:**

- Python
- Neural Network Framework (TensorFlow)
- Raspberry Pi
- Camera Module (for image processing)
- Motor Control System
- Sensors for Environmental Feedback

![Untitled](Project-ITT-Neural-Network-autonomous-movement%20b7c62fb4862b4269bd97cda4906d18eb/Untitled.png)

In this first step all the data that will be used later in the training stage is collected, in
this case the data are images accompanied by the information of the defined direction
to which the chair is going to move depending on the route that is being executed. More
specifically it can be added that a folder with images and a record file is created for
each run that is made with the path that is being learned. The result would be something
like this:

![Untitled](Project-ITT-Neural-Network-autonomous-movement%20b7c62fb4862b4269bd97cda4906d18eb/Untitled%201.png)

It is important to clarify that the address data must be numeric for this reason the
following distribution was selected:

![Untitled](Project-ITT-Neural-Network-autonomous-movement%20b7c62fb4862b4269bd97cda4906d18eb/Untitled%202.png)

![Untitled](Project-ITT-Neural-Network-autonomous-movement%20b7c62fb4862b4269bd97cda4906d18eb/Untitled%203.png)

This step is very critical, since in this the training of the model is carried out that will
make the predictions later, as can be seen in the previous figure is takes the data
collected in the first step and through the neural network, implements gives mainly
with the library of Tensorflow in Python, the file of the model that guides the
wheelchair to complete the autonomous route.

![Untitled](Project-ITT-Neural-Network-autonomous-movement%20b7c62fb4862b4269bd97cda4906d18eb/Untitled%204.png)

![Untitled](Project-ITT-Neural-Network-autonomous-movement%20b7c62fb4862b4269bd97cda4906d18eb/Untitled%205.png)

In these figures you can see the data of the directions of an example route that was
trained using this methodology, as you can see this route only has two directions that
are forward and to the right. Another important aspect to note is that the process of
balancing the data, which prevents bias towards any direction, was equal to the
information originally captured, this because no address has a significant domain with
respect to the others.

![Untitled](Project-ITT-Neural-Network-autonomous-movement%20b7c62fb4862b4269bd97cda4906d18eb/Untitled%206.png)

The Epoch means training the neural network with all the training data for a cycle. In
an Epoch, all the data is used exactly once. The loss is calculated on the basis of training
and validation and its interpretation is how well the model is performing for these two
sets. The lower the loss, the better the model, in the graph you can see that as the Epochs
progress, the more the loss decreases, which is an indicator that the model has been
trained in the right way.

![Untitled](Project-ITT-Neural-Network-autonomous-movement%20b7c62fb4862b4269bd97cda4906d18eb/Untitled%207.png)

Finally, the final step would be to implement the model, for this the model and an image
are loaded, the model is given the image as input and it reports as output the prediction
numerically. Subsequently, this data must be analyzed and processed by the main
program of the project to make the movements.

![Untitled](Project-ITT-Neural-Network-autonomous-movement%20b7c62fb4862b4269bd97cda4906d18eb/Untitled%208.png)

Example result indicating that the "to the right" command must be
executed.

**Conclusion:**

Although the main goal of this project was to implement autonomous movements that use TensorFlow’s neural network algorithm in Python to construct the archetypes of each route, and save them, so that the next time you want to do the same route, you only need to load the route model with the user friendly interface, due to the time limit and the difficulty of the task this feature was left out in the final version presented to the judges. However, with this project and the knowledge learned from this experience, it could be possible to continue when we leave it and finish this interesting feature of the intelligent wheelchair.