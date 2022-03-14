## Attack Guide

* Make several copies of ```fastrnn.py``` and ```fastgrnn.py```, each with different values of ```eps```.
* Launch through terminal (e.g., ```python fastrnn.py```).
* We do not use a single notebook to perform the attack for all values of epsilon. This is because due to not using eager execution, the call to the model graph becomes slower with each call. 
