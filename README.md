# Color-Blindness-Tool # OpenCV
This is a python tool which takes a video as an input and identify on which frames there are pixel which a color blind person cannot identify.

Basically there are three types of color blindness in the world -
1) Protanopia (red-blind)
2) Deuteranopia (green-blind)
3) Tritanpoia (blue-blind)

With this tool user can identify if a particular video is fit for given type of color blind person.


Typical command line calls might look like::

python colorBlind.py <videofile> <type of color-Blindness (Proto | Deuto | Trita)>

There is a optional parameter --ssim which can help you to control over number of frames you want to process.
