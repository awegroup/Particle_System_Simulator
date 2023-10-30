// Gmsh project created on Sun Oct 29 16:10:43 2023
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {-0.5, -0.5, 1, 1, 1, 0};
//+
Rectangle(2) = {-.5, -0.5, 0, 1, 1, 0};
//+
Curve Loop(3) = {1, 2, 3, 4};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {5, 6, 7, 8};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {2, 3, 4, 1};
//+
Curve Loop(6) = {6, 7, 8, 5};
//+
Curve Loop(7) = {6, 7, 8, 5};
//+
Curve Loop(8) = {6, 7, 8, 5};
//+
Curve Loop(9) = {6, 7, 8, 5};
//+
Curve Loop(10) = {6, 7, 8, 5};
//+
Curve Loop(11) = {2, 3, 4, 1};
//+
Curve Loop(12) = {6, 7, 8, 5};
//+
Plane Surface(5) = {11, 12};
