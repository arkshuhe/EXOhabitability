# Search for the habitable exoplanets
#### Utilize the machine learning algorithms to look for habitable exoplanets

### Motivation:
Several machine learning algorithms have been used to predict the habitability of known exoplanets, it includes: deep learning, anomaly detection etc. It would be encouraging to explore other ways to achieve the similiar goals.

PHL is an prestigous institue on the domain of habitability of exoplanets, which has published and updated the list of potential habitable exoplanets time to time, and even the models it has utilized are sophisticate and improved constantly, it is still possibility of FALSE NEGATIVE ( Habitable in fact, but inhabitable as predicted). Thus, I am trying to look for such FALSE NEGATIVE as assisted with Machine Learning Alorithms, specifically clustering.

### Approach:

1. Pick up the typical features of exoplanets for the detection of habitability from PHL.
2. Apply with 3 ways consisted of ML algorithms to find out the inhabitable candidates which are most closed to habitable exoplanets based on the results of PHL.
3. Scrutize the candidates by ChatGPT-4 and the professional website on the study of habitability of exoplanets to lock down the FALSE-NEGATIVE candidates.

### Conclusion:
1. One candidate was detected and it could be habitable even it is judged as inhabitable in PHL.
2. The whole process does NOT mainly take the comprehensive criterias of habitability into consideration, even some criterial factors such as disregarding the impact from planetary systems, star, water. It should be considered in the next.
3. Even in the clustering algorithms, the weights of features have been treated as equal, which should be reconsidered.

### Next steps:
1. The new algorithms are required to be built to fully consider the comprehensive criterias of habitability.
2. The features of habitability needs to be weighted.

### Notes:
1. It is required to input OPENAI API KEY and Google SEARCH API KEY.
2. The whole propgrams is consisted of 2 sections:
   - main program: exoplanets_confirmation.ipynb
   - handy functions: helper.py
