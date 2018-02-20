# A standalone version of DBOW3

Only depends on OpenCV. 
Code borrowed from [https://github.com/rmsalinas/dbow3](https://github.com/rmsalinas/dbow3). 

This is a standalone demo to use DBOW ( bag-of-words ). 

## Dependencies 
- Download the ORB-vocabulary from [https://github.com/rmsalinas/DBow3/blob/master/orbvoc.dbow3](https://github.com/rmsalinas/DBow3/blob/master/orbvoc.dbow3)
- Test dataset, say [New College](http://www.robots.ox.ac.uk/NewCollegeData/)


## How to compile 
```
mkdir build 
cmake .. 
make 
```

## dbow3\_standalone.cpp

    1. Read Image
    2. Extract ORB keypoints and descriptor
    3. Add to DB
    4. Query from DB using current
    5. GeometricVerify loop candidates
    6. go to step 1

