# system-identification-R7015E

Lab works from the LTU course System Identification R7015E. 


## In this directory:

- [Lab1: Kalman vs. Luenberger][] 

   - Three predictor filters are studied and compared for a total of four datasets, three of which are synthetic and the last extracted from a real system. The filters under study are a Luenberg Observer, a Time Variant Kalman Filter and a Time Invariant one. The potential of these filters to make one-step ahead predictions is noted, and its tuning capabilities explored. Learned how to implement these three predictors and experimented with a Monte Carlo approach that appeared successful at tuning Kalman filters. 


- [Lab2: Successes and pitfalls of frequency-domain identification methods][]
  
  - For a given dataset three techniques of system identification were put to test: Correlation Analysis, empyrical transfer function estimate (ETFE) and smoothing of the ETFE. It is found that it is easier to get an accurate low-order model with the correlation analysis than with the pure frequency domain techniques for this case.


- [Lab3: ML-based (maximum likelihood) estimation of occupancy patterns][]

  - For the problem of estimating occupancy levels in a room, a ML estimator is used for identifying an ARX model for the CO2 level. The likelihood function of the CO2 level is derived as the likelihood of the noise present in the measurements, from which a maximum likelihood estimator is computed by using numerical constrained optimization tools for a training dataset. Then, the ARX model is used to obtain a prediction of the occupancy for a testing dataset. The performance of the model is then evaluated and deemed satisfactory. The most significant learning outcomes were the derivation and understanding of the likelihood function, the computing of the maximum likelihood estimator for a particular set of data and its application to a real world problem, where the model for the desired predicted magnitude was not explicitly provided.


- [Lab4: Identification of the engine requirements for a wheel loader][]
  - This work aims to find a model for the engine speed of a wheel loader using a small dataset of time-series data of various variables (features), not all of which may be relevant towards predicting the required engine speed. In the end, it turns out that a model with very few input features can still produce a significantly representative prediction model. Learned how to approach a black-box identification problem and use MATLAB-implemented tools - both using a graphical user interface (GUI) toolbox and the command-line interface (CLI) - to solve it, as well as perform basic dimensionality reduction tasks.


[Lab1: Kalman vs. Luenberger]:                                             https://github.com/pabsan-0/R7015E-system-identification/tree/main/Lab1
[Lab2: Successes and pitfalls of frequency-domain identification methods]: https://github.com/pabsan-0/R7015E-system-identification/tree/main/Lab2
[Lab3: ML-based (maximum likelihood) estimation of occupancy patterns]:    https://github.com/pabsan-0/R7015E-system-identification/tree/main/Lab3
[Lab4: Identification of the engine requirements for a wheel loader]:      https://github.com/pabsan-0/R7015E-system-identification/tree/main/Lab4
