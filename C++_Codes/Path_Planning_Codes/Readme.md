To implement any Path Planning algorithm: -
1)  Change the paths inside the codes for Maps, Coordinates and Path file output
2)  build the fils with the line-
g++ -std=c++0x -ggdb `pkg-config --cflags opencv` -o `basename Astar.cpp .cpp` Astar.cpp `pkg-config --libs opencv`
3) Run in Command bar as -
./Algo_Name Map_Number
Example
./RRT 0
./Astar 2
