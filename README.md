# pytorch-inference

### Purpose
I realize that including all of pytorch's functionality in an OpenCL implementation
is difficult for various reasons. However, the fact remains that an OpenCL
runtime would be quite useful. For this reason, I've done quite a bit of work
to try and write functions using ArrayFire that mimic pytorch functions exactly - 
which allows us to use the pytorch-trained weights in our C++ program.

### Dependencies
```
cmake >= 3.5
pytorch >= 0.1.9
arrayfire >= 3.4 (?)
Py_Cpp (python3 port already included)
```

### Contributions
I welcome most anything. I would especially welcome help from those who know more about 
ArrayFire and/or pytorch who can help optimize, suggest improvements, help out with 
documentation, anything/everything.

I would ask that you follow these steps for organization:
1. Open an issue with proposed changes (tag appropriately)
2. Fork the repo and add a branch for the changes
3. Be patient - I'll get to it as soon as I can


Note on the licensing - I believe in open-source software and I like the GPL, but if 
someone has a good reason to change the license please let me know and I'll consider it.

Obligatory note: I am NOT promoting ArrayFire or using its name to promote this project 
in any way. If someone from ArrayFire has issues with this they should please let me
know promptly and I'll do my best to rectify the situation. I'm not distributing the source
code or binary so I hope there aren't any conflicts.