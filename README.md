# FieldSolver
Code examples how to solve possion equation efficienty in cartesian and cylindrical coordinates.

There are many very sophisticated codes which allow the user to calculate electric or magnetic fields e.g. fipy but the problem I encountered was that these codes are very complex. They use efficient but complex methods, e.g. finite volume what makes it hard to extend or use them for specialized problems.

In physics, most problems differ extremly from each other and for solving these problems one needs solutions which are efficient to some degree, but simplicity is more important to understand the code and extend it further for the special problem one might have. 

The field solvers provided here work for an evenly spaced grid, take simple bitmaps as input and will calculate the electric and magnetic field so it can be included in your work.
