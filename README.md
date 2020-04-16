# Interpretable COVID prediction with Semi-Supervised NMF 
<p align="center">
<b>About this repository</b>
<br>
</p>
<p>
This repository contains scripts used to learn interpretable features from X-Ray imaging data originating from various sources related to the COVID-19 pandemic. 
</p>

<br>
<p align="center">
<b>Files Contained</b>
<br>
</p>

<p>
<ul style="list-style-type:circle">
  <li>Extract_COVIDnet_Features.py - Utilizes the pretrained COVID-Net models and extracts the learned representations of the data at the final 3 layers of COVID-Net.  The header of this script contains information on how to run it. </li>
  <li>kmeans_DenseNet_Representations.m - Performs k-means clustering on the learned representation of the data for DenseNet [Add Reference], and computes the average purity of clusters over each different initialization of k-means.  </li>
  <li>kmeans_COVIDnet_Representations.m -  Performs k-means clustering on the learned representations of the data for COVID-Net and computes the average purity of clusters over each different initialization of k-means.</li>
  <li>make_COVIDx_labels.py - Reads in test_COVIDx.txt and train_COVIDx.txt and creates .mat files for the labels</li>
</ul>
</p>

<br>
<p align="center">
<b>Requirements</b>
<br>
</p>

<p>
  <ul>
    <li> To generate the data set for COVID-Net and use COVID-Net - PyDicom, Pandas, Jupyter, Tensorflow 1.15, OpenCV 4.2.0, Python 3.5+, Numpy, Scikit-Learn, Matplotlib </li>
  </ul>
</p>


<br>
<p align="center">
<b>Usage</b>
<br>
</p>

<p>
To perform clustering on the learned representations of COVID-Net:
  <ol>
    <li> Clone the repository at https://github.com/lindawangg/COVID-Net and follow their instructions on how to generate the relevant data set </li>
    <li> Run the Extract_COVIDnet_Features.py script by providing as arguments the path to the pretrained model/data </li>
    <li> Run the script make_COVIDx_labels.py to generate .mat files containing the numerical labels for each data point.  This is needed when computing the average purity for k-means clustering.</li>
    <li> Run the Octave (MATLAB) script kmeans_COVIDnet_Representations.m after adjusting the load paths and variables as appropriate to the .mat files generated in step 2 and 3.  Warning: The 4/15/2020 update of COVID-Net changed the number and type of classes being used in the classification task.</li>
  </ol>
</p>

<!--
<br>
<p align="center">
<b>List of all contributors</b>
<br>
</p>

<p>
  <ul style="list-style-type:circle">
    <li>jhaddock@math.ucla.edu</li>
    <li>ksmill327@gmail.com</li>
    <li>alona.kryshchenko@csuci.edu</li>
    <li>kleonard.ci@gmail.com</li>
    <li>es5223@nyu.edu</li>
    <li>cwang27@ua.edu</li>
    <li>rachel.grotheer@goucher.edu</li>
    <li>psalanevich@math.ucla.edu</li>
    <li>yotamya@math.ucla.edu</li>
    <li>wenli@math.ucla.edu</li>
    <li>chu@math.ucla.edu</li>
    <li>shaydeu@math.ucla.edu</li>
    <li>mijuahn@gmail.com</li>
    <li>madushani67@gmail.com</li>
    <li>nerutt@gmail.com</li>
    <li>lara.kassab@colostate.edu</li>
    <li>tmerkh@g.ucla.edu*</li>
  </ul>
</p>

<p>
  *-contact for the page
</p>
-->
