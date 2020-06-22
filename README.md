## Palm And Face Regions Detection
### Using opencv + python
<ol>
<li> after cloning repo cd into it - "cd image-processing".</li>
<li> then run "python -m venv venv/" to initilize python virtual environment.</li> 
<li> activate env by running "source venv/bin/activate".</li>
<li> run "pip install -r requirements.txt" to install python dependencies.</li>
<li> run "activate" again. </li>
</ol>

### Run the skinColorDetectionHSV.py for detection algorithm    

### skinColorDetectionHSV.py
<p>Includes the HSV color implementation. You can contour tweak parameters using trackbar</p>

### histogramsHSV.py
<p>Includes the histogram of the 101th frame to HSV color implementation. A mask is applied to filter out skin pixels</p>

### skinColorDetectionYcrcb.py
<p>Includes the Ycbcr color implementation. The frame rate is quit slow because of using some custom made functions</p>


