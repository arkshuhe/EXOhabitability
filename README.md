# Search for the habitable exoplanets
#### Utilize the machine learning algorithms to look for habitable exoplanets

### Motivation:

Various machine learning algorithms, including deep learning and anomaly detection, have been employed to assess the habitability of known exoplanets. Exploring alternative methods to achieve similar objectives would be beneficial. 

The Planetary Habitability Laboratory (PHL), a prestigious institute in the field of exoplanet habitability, regularly publishes and updates a list of potentially habitable exoplanets. Despite the sophistication and continual improvement of their models, there remains the possibility of false negatives (planets that are habitable but predicted as uninhabitable). My goal is to identify such false negatives using machine learning algorithms, with a focus on clustering.

### Approach:

1. Identify key exoplanet features for habitability detection from PHL's data.
2. Employ three different machine learning approaches to pinpoint candidates that are nearly habitable based on PHL's findings.
3. Examine these candidates through ChatGPT-4 and reputable websites dedicated to the study of exoplanet habitability to confirm false-negative cases.

### Conclusion:

One candidate has been identified that could be habitable, although it was deemed uninhabitable by PHL. This process did not primarily consider a comprehensive set of habitability criteria, omitting key factors such as the impact of planetary systems, stars, and the presence of water. 

Additionally, in the clustering algorithms, all feature weights were treated equally, which may need revision.

### Next steps:

- Develop new algorithms that fully incorporate comprehensive criteria for habitability.
- Assign appropriate weights to different habitability features.

### Notes:
1. It is required to input OPENAI API KEY and Google SEARCH API KEY.
2. The whole propgrams is consisted of 2 sections:
   - main program: exoplanets_confirmation.ipynb
   - handy functions: helper.py
