#**x:pression**

## Synopsis

This project was fully written in C++11 early 2016, using the CLM-framework to VR/AR course at CIn UFPE. This project is a computer vision algorithm that makes a real time tracking of the user's face and detect the actual emotion based on the extracted features. It was my first experience with a face detection algorithm.

## About the project

In this project we use the [CLM-framework](github.com/TadasBaltrusaitis/CLM-framework) to make the face detection and then we calibrate the algorithm for the user's face. This calibration stage is needed because i was not able to create a big database to teach my algorithm about human expressions. But after each calibration the algorithm writes in a excel sheet all the calibration data, so the database is always being populated and keeps growing at each calibration and can be reused everytime when someone runs the program. After calibrate the algorithm, he will detect your expressions in real-time and generate a excel sheet that informs yours expression along the time.

## Installation

You only need to install the [Visual Studio IDE](www.visualstudio.com) and open the solution.

## Contributors

This project was developed with the help of my friend @fmm4

## Contact
eduardohmrodrigues@gmail.com or ehmr@cin.ufpe.br