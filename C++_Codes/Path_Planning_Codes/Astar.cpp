#include <ctype.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <numeric>
#include <math.h>
#include <cmath>
#include <utility>
#include <map>
#include <algorithm>
#include <unordered_map>
#include <time.h>

#define HEIGHT 2048
#define LENGTH 2048

using namespace std;
using namespace cv;
struct passwd *pw = getpwuid(getuid());
const string homedir = pw->pw_dir;
string coordpath = homedir + "/Desktop/Akash/Path_planning/Generator/LooseFiles/Maps/coord";
string outpath = homedir + "/Desktop/Akash/Path_planning/Generator/LooseFiles/Paths_Astar/path";
vector<int> start = {(1024), (1024)};
vector<int> stop = {(0), (0)};
uint8_t m [2048][2048];
int x = 0;
int y = 0;
int X = 2048;
int Y = 2048;
double scale = 19.93970181650108;
int mvp = 255;
int nmvp = 0;
int obs = 0;
int wal = 150;
int walhigh = 3;
int wallow = 2;
Mat img;
vector<vector<int>> nei;
string mappath = homedir + "/Desktop/Akash/Path_planning/Generator/LooseFiles/Maps/map";

struct idmap
{
	int pos;
	double Value;
};

////////////////////////////////////////////////////////////////////////////////
// Name : read_coord	                                                        
// Return type : void	                                                        
// Parameters : int &, int &						                            
// Function : Reads the coordinates from the file coord.txt					 	
// Return : N/A  																
////////////////////////////////////////////////////////////////////////////////	
void read_coord(int &x, int &y) {
	string line;
	int i = 0;
	int neg_flag = 0;
	ifstream myfile (coordpath.c_str());
	if (myfile.is_open()) {
		getline (myfile,line);
		getline (myfile,line);
		getline (myfile,line);
	}
	line.erase(line.begin() + 0, line.begin() + 17);
	while(line[i] != ',') {
		if(line[i] == '-') {
			neg_flag = -1;
		}
		else {
			x = (x*10)+(line[i] - 48);
		}
		i++;
	}
	if(neg_flag)
		x *= -1;
	i+=2;
	neg_flag = 0;
	while(line[i]) {
		if(line[i] == '-') {
			neg_flag = -1;
		}
		else {
			y = (y*10)+line[i] - 48;
		}
		i++;
	}
	if(neg_flag)
		y *= -1;
	myfile.close();
}

/**  @function Erosion  */
void Erosion(Mat& image)
{ 
  int erosion_elem = 0;
  int erosion_size = 10;
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

  /// Apply the erosion operation
  erode( image, image, element );
  // imshow( "Erosion Demo", image );
  // waitKey(-1); 


}

////////////////////////////////////////////////////////////////////////////////
// Name : read_map		                                                        
// Return type : void	                                                        
// Parameters : void						            		                
// Function : Reads the map from the file map.png							 	
// Return : N/A  																
////////////////////////////////////////////////////////////////////////////////	
void read_map() {
	img = imread(mappath.c_str(), 0);
	img.convertTo(img, CV_8U);
    Erosion(img);
    for(int i = 0; i<2048;i++){
    	for(int j = 0; j<2048; j++){
    		m[i][j] = (int)img.at<uint8_t>(i,j);
    	}
    }
}

/////////////////////////////////////////////////////////////////////////////////
//                          Function Definition
//   Name  - DisplayPath
//   Function  - used to diaply path if necessary
//   Arguments  - vector<vector<int>> Line
/////////////////////////////////////////////////////////////////////////////////

void DisplayPath(vector<vector<int>> path) {
		for (int i = 0; i < path.size(); i ++){
			m[path[i][0]][path[i][1]] = 50;
		}
		Mat dataMatrix1(2048,2048,CV_8UC1, m);
        //namedWindow( "Display window", WINDOW_AUTOSIZE );
        namedWindow("Display window",CV_WINDOW_NORMAL);
		imshow( "Display window", dataMatrix1 );   // Show our image inside it.
    	waitKey(-1); 
    	for (int i = 0; i < path.size(); i ++){
			m[path[i][0]][path[i][1]] = 255;
		}
}

////////////////////////////////////////////////////////////////////////////////
// Name : rot90		                                                        
// Return type : void	                                                        
// Parameters : int num (number of rotations)						            		                
// Function : Rotates the map							 	
// Return : N/A  																
////////////////////////////////////////////////////////////////////////////////	
void rot90(int num) {
	int temp;
	for (int i = 0; i<num;i++) {
		for (int n = 0; n< (2048 - 2); n++){
			for(int l = n+1; l<(2048 - 1); l++){
				temp = m[n][l];
				m[n][l] = m[l][n];
				m[l][n] = temp;
			}
		}
	}
}
  	
////////////////////////////////////////////////////////////////////////////////
// Name : fliplr		                                                        
// Return type : void	                                                        
// Parameters : N/A						            		                
// Function : flips the map horizontlly							 	
// Return : N/A  																
////////////////////////////////////////////////////////////////////////////////	
void fliplr() {
	int temp;
	for (int n = 0; n< (2048 ); n++){
		for(int l = 0; l<(2048/2); l++){
			temp = m[n][l];
			m[n][l] = m[n][2048 - l];
			m[n][2048 - l] = temp;
		}
	}
}
  

////////////////////////////////////////////////////////////////////////////////
//                          Function Definition									
//   Name  - Neighbors                                                          
//   Function  - Finds the neighboring points around a given point in a grid.	
//               Depending on the requirement, it is used for finding all the 	
//               points between the high and low x, y distance. And it is also 	
//               used to find a points for the search step. For 3, returns		
//               every 10th neighbor to prevent over-crowding of branches.		
//   Arguments  - int ch -> 1(low 3 and high 4) - Check if Valid move,				
//                     2(low 1 and high 2) - Check for noise					
//                     3(low 20 and high 21) - Only consider perimeter for jump	
//                int ptx, int pty -> point of reference								
////////////////////////////////////////////////////////////////////////////////

	
int Neighbors(int ch , int ptx, int pty) {
	nei.clear();
	int low, high;
    if (ch == 1) {
	    low = 4;
    	high = 5;
    }
    if (ch == 2) {
        low = 10;
        high = 10;
    }
    if (ch == 5) {
        low = 1;
        high = 2;
    }

    if (ch == 3) {
    	low = 15;
        high = 16;
        int thres = 13;
        for (int dx=-low; dx<=high; dx++)
	  		for (int dy=-low; dy<=high; dy++) 
			    if (dx || dy){
			        int x = ptx+dx, y=pty+dy;
	   		        if ((x >= 0) && (x < X) && (y >= 0) && (y < Y) && 
	   		        	((abs(x-ptx)>thres) || (abs(y-pty)>thres))) {
			            nei.push_back({x,y});
		        }
		    }
	return 0;
	}
    if (ch == 4) {
        low = wallow;
        high = walhigh;
	}

	for (int dx=-low; dx<=high; dx++)
  		for (int dy=-low; dy<=high; dy++) 
		    if (dx || dy){
		        int x = ptx+dx, y=pty+dy;
   		        if (x >= 0 && x < X && y >= 0 && y < Y){
		            nei.push_back({x,y});

	        }
	    }
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//                          Function Definition									
//   Name  - ExtendStarStopt													
//   Function  - Extend the walls in the map to give enough moving room for the 
//               Search algorithm. Done because Gazebo gives coordinates of		
//               Start and Stop right next to the walls 						
//   Arguments  - N/A 															
////////////////////////////////////////////////////////////////////////////////

void ExtendStartStop() {
    //cout << "Extending start ..." << endl;
    Neighbors(2,start[0],start[1]);
    for( int i = 0; i<nei.size(); i++)
       	m[nei[i][0]][nei[i][1]] = mvp;
    //cout << "Extending stop ..." << endl;
 	Neighbors(2,stop[0],stop[1]);
    for( int i = 0; i<nei.size(); i++)
       	m[nei[i][0]][nei[i][1]] = mvp;
}


////////////////////////////////////////////////////////////////////////////////
//                          Function Definition									
//   Name  - ang 																
//   Function  - Used to find angle between two lines 							
//   Arguments  - vector<vector<int>> lineA, vector<vector<int>> lineB 													
////////////////////////////////////////////////////////////////////////////////

double ang(vector<vector<int>> lineA, vector<vector<int>> lineB) {
    
    // Get nicer vector form
    int vA[] = {(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])};
    int vB[] = {(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])};
    
    // Get dot prod
    double dot_prod = inner_product(begin(vA), end(vA), begin(vB), 0.0);

    // Get magnitudes
    double magA = pow(inner_product(begin(vA), end(vA), begin(vA), 0.0),0.5);
    double magB = pow(inner_product(begin(vB), end(vB), begin(vB), 0.0),0.5);
    double angle;
    try {

        // Get angle in radians and then convert to degrees
         angle = acos(dot_prod/magB/magA);
    }
    catch(...) {
         angle = 0;
    }

    // Basically doing angle <- angle mod 360
    double ang_deg = fmod(((180.0/3.14)*(angle)),360.0);
    
    if (std::isnan(ang_deg)) {
        return 180;
    }
    return ang_deg;
}

////////////////////////////////////////////////////////////////////////////////	
//                          Function Definition 								
//   Name  - get_line															
//   Function  - Returns all the points in a grid that fall between two given	
//               referrence points. 											
//   Arguments  - int x1, int y1 -> (pt1), int x2, int y2 -> (pt2) 								
////////////////////////////////////////////////////////////////////////////////

vector<vector<int>> get_line(int x1, int y1, int x2, int y2) {
    vector<vector<int>> points;
    bool issteep;
    if(abs(y2-y1) > abs(x2-x1))
        issteep = true;
    else
        issteep = false;
    if (issteep){
        int temp;
        temp = x1;
        x1 = y1;
        y1 = temp;
        temp = x2;
        x2 = y2;
        y2 = temp;
    }
    bool rev = false;
    if (x1 > x2){
        int temp;
        temp = x1;
        x1 = x2;
        x2 = temp;
        temp = y1;
        y1 = y2;
        y2 = temp;
        rev = true;
    }
    int deltax = x2 - x1;
    int deltay = abs(y2-y1);
    int error = (int)(deltax / 2);
    int y = y1;
    int ystep = 0;
    if (y1 < y2)
        ystep = 1;
    else
        ystep = -1;
    int count = 0;
    for (int x = x1; x<x2+1 ; x++){
        if (issteep && (count%4 == 0)){
            points.push_back({y, x});
        }
        else if (count%4 == 0){
            points.push_back({x, y});
        }
        error -= deltay;
        if (error < 0){
            y += ystep;
            error += deltax;
        }
        count++;
    }

    // Reverse the list if the coordinates were reversed
    if (rev)
        reverse(points.begin(), points.end());
    return points;
}


////////////////////////////////////////////////////////////////////////////////
//                          Function Definition									
//   Name  - CheckValid															
//   Function  - Checks if a move is valid between two given referrence points.	
//               The condition is that the two points should not be non movable 
//               points and every point between the two should have a buffer 	
//               from the wall. The buffer if found using Neighbors with x = 1.	
//               Returns 1 for valid move, and 0 for invalid move.				
//   Arguments  - vector<int> pt1, vector<int> pt2														
////////////////////////////////////////////////////////////////////////////////
int CheckValid(vector<int> pt1, vector<int> pt2) {
	vector<vector<int>> gline;
	gline = get_line(pt1[0],pt1[1],pt2[0],pt2[1]);
    for(int i = 0;i<gline.size();i++){
        if ((int)m[gline[i][0]][gline[i][1]] == nmvp){
            return 0;
        }
    }
    return 1;
}

////////////////////////////////////////////////////////////////////////////////
//                          Function Definition									
//   Name  - CutNeighbors														
//   Function  - Returns Valid movable points from neighbors   					
//   Arguments  - pair<int,int> pt(point of referrence), map<pair<int,int>, 
//				  vector<vector<int>>> sD(Dictionary of the branches of the other tree.), 
//				  pair<int,int> ap(Previos point for reference)												
////////////////////////////////////////////////////////////////////////////////
    
vector<vector<int>> CutNeighbors(pair<int,int> pos, vector<pair<int,int>> sD, pair<int,int> ap) {
    vector<vector<int>> nv;
    int i;
    vector<vector<int>> neiC = nei;
    for (i = 0; i < neiC.size(); i++) {
    	vector<vector<int>> lineA = {{ap.first, ap.second},{pos.first, pos.second}};
		vector<vector<int>> lineB = {{pos.first, pos.second},{neiC[i][0],neiC[i][1]}};
		int f = neiC[i][0];
		int s = neiC[i][1];
		int it = -1;
		for (int k = 0; k<sD.size(); k++){
			if((sD[k].first == f) && (sD[k].second == s)){
				it = k;
				break;
			}
		}
        if ((CheckValid({pos.first, pos.second},neiC[i]) == 1) && 
        (ang(lineA,lineB) <=90.0 )&&
        (it == -1))
            nv.push_back({neiC[i][0], neiC[i][1]}); 
    }
    return nv;
}

/////////////////////////////////////////////////////////////////////////////////
//                          Function Definition
//   Name  - LineComp
//   Function  - This Function adds all intermediate points to the Line for 
//               easier movement control by the car
//   Arguments  - vector<vector<int>> Line -> Line with scattered points instead of being continuous
/////////////////////////////////////////////////////////////////////////////////

vector<vector<int>> LineComp(vector<vector<int>> Line) {
    vector<vector<int>> NewLine, gline; 
    for (int i = 0; i<Line.size()-1; i++){
    	gline = get_line(Line[i][0],Line[i][1],Line[i+1][0],Line[i+1][1]);
    	for (int j = 0; j<gline.size()-1; j++){
        	NewLine.push_back({gline[j][0],gline[j][1]});
        }
    }
    return NewLine;
}

////////////////////////////////////////////////////////////////////////////////
//                          Function Definition
//   Name  - LineOptimization
//   Function  - This function optimizes the line by removing redundant points. 
//               It does this by checking every point on the line with all the
//               points after it to check for line of sight. If found, it 
//               removes all the intermediate points.   
//   Arguments  - vector<vector<int>> Line1 -> Final Path
/////////////////////////////////////////////////////////////////////////////////

vector<vector<int>> LineOptimization(vector<vector<int>> Line1) {
    
    // Variable declaration for line optimization //
    vector<vector<int>> Line2;
    int pt1 = 0;
    int pt2 = Line1.size()-1;
    //cout <<"Length of path Before = " << Line1.size() << endl;
    
    // Line Optimization //
    while (pt1<(Line1.size()-1) && pt2>pt1) {
        while ((CheckValid(Line1[pt1],Line1[pt2]) != 1) && 
        (pt2 >pt1)){
        	pt2 -=1;
        }
        Line2.push_back(Line1[pt1]);
        pt1 = pt2+1;
        pt2 = Line1.size()-1;
    }
    Line2.push_back(Line1[Line1.size()-1]);
    //cout << "Useful points = " << Line2.size() << endl;
    Line2 = LineComp(Line2);
    //cout << "Length of path After = " << Line2.size() << endl;
    vector<vector<int>> Line3;
    pt2 = 0;
    pt1 = Line2.size()-1;
    //cout <<"Length of path Before(reverse) = " << Line2.size() << endl;
    
    // Line Optimization //
    while ((pt1 > 0) && (pt1>pt2)) {
        while ((CheckValid(Line2[pt2],Line2[pt1]) != 1) && 
        (pt1 > pt2)){
        	//cout << (CheckValid(Line2[pt2],Line2[pt1])) << " " << Line2[pt2][0] << " " << Line2[pt2][1]<< " " << Line2[pt1][0]<< " " << Line2[pt1][1]<< endl; 
        	pt2 +=1;
        }
        //cout << "\n"<< Line2[pt1][0] << " " <<Line2[pt1][1] << "\n\n"; 
        Line3.push_back(Line2[pt1]);
        pt1 = pt2-1;
        pt2 = 0;
    }
    Line3.push_back(Line2[0]);
    //cout << "Useful points(reverse) = " << Line3.size() << endl;
    Line3 = LineComp(Line3);
    //cout << "Length of path After(reverse) = " << Line3.size() << endl;
    return Line3;
}

////////////////////////////////////////////////////////////////////////////////
//                          Function Definition
//   Name  - dot
//   Function  - Used to find dot product between two vectors
//   Arguments  - vector<int> vA -> Vector A, vector<int> vB -> Vector B
////////////////////////////////////////////////////////////////////////////////

int dot(vector<int> vA, vector<int> vB){
    return vA[0]*vB[0]+vA[1]*vB[1];
}

/////////////////////////////////////////////////////////////////////////////////
//                          Function Definition
//   Name  - AngleDef
//   Function  - Used to create a list of all the angular changes in the entire
//               path. It does this by finding the angular difference at each 
//               position in the map w.r.t the next position
//   Arguments  - vector<vector<int>> Line
/////////////////////////////////////////////////////////////////////////////////

vector<idmap> AngleDef(vector<vector<int>> Line){
    idmap temp2;
    vector<idmap> anglist ;
    vector<vector<int>> AF = LineComp(Line);
    vector<int> temp;
    for (int i = 1; i<Line.size() - 1; i++){
    	vector<vector<int>> lineA = {{Line[i-1][0], Line[i-1][1]},{Line[i][0], Line[i][1]}};
		vector<vector<int>> lineB = {{Line[i][0],Line[i][1]},{Line[i+1][0], Line[i+1][1]}};
    	temp = {Line[i][0], Line[i][1]};
    	int it = distance(AF.begin(), find(AF.begin(), AF.end(), temp));
        temp2.pos = it;
        temp2.Value = ang({Line[i-1],Line[i]},{Line[i],Line[i+1]});
        anglist.push_back(temp2);
    }
    return anglist;
}

/////////////////////////////////////////////////////////////////////////////////
//                          Function Definition
//   Name  - SpeedList
//   Function  - Calculates the speeds at which the car has to travel
//   Arguments  - vector<vector<int>> Line, map<int, double> angleList
/////////////////////////////////////////////////////////////////////////////////


vector<double> SpeedList(vector<vector<int>> Line, vector<idmap> angleList) {
    vector<double> speedlist;
    double speed;
    int i;
    for (i = 0; i < Line.size() - 20; i++) {
        int flag = 0;
        for (int j = 0; j< angleList.size(); j++){
            double minspeed = 100.0 - int((angleList[j].Value/45.0) * 70.0);
            
            // Change ranges depending on use case //
            if ((angleList[j].pos - i) == 0) {
                speed = minspeed;
                speedlist.push_back(speed);
                flag = 1;                
                break;
            }
            else if (((angleList[j].pos - i) < 20) && ((angleList[j].pos - i) > 0)) {
                speed = 100.0 - ((100.0 - minspeed)*(1 - ((angleList[j].pos - i)/20.0)));
                speedlist.push_back(speed);
                flag = 1;
                break;
            }
            else if (((angleList[j].pos - i)> -10) && ((angleList[j].pos - i) < 0)) {
                speed = 100.0 - ((100.0 - minspeed)*(1 + ((angleList[j].pos - i)/10.0)));
                speedlist.push_back(speed);
                flag = 1;
                break;
            }
        }
        if (flag == 0) 
            speedlist.push_back(100.0);
        
    }
    for(int k = 0; k<19; k++)
        speedlist.push_back((100.0 - (100.0*(k/19.0))));
       
    return speedlist;
}

/////////////////////////////////////////////////////////////////////////////////
//                          Function Definition
//   Name  - Heuristic(Manhatten Distance)
//   Function  - Returns Distance between two coordinates 
//   Arguments  - vector<int> pt1, vector<int> pt2
/////////////////////////////////////////////////////////////////////////////////

double Heuristic(vector<int> pt1, vector<int> pt2) {
    return(sqrt(((pt1[0]-pt2[0])*(pt1[0]-pt2[0])) + ((pt1[1]-pt2[1])*(pt1[1]-pt2[1]))));
}

/////////////////////////////////////////////////////////////////////////////////
//                          Function Definition
//   Name  - Dist
//   Function  - Returns Distance between two coordinates (x1-x2) + (y1+y2) 
//   Arguments  - vector<int> pt1, vector<int> pt2
/////////////////////////////////////////////////////////////////////////////////

double Dist(vector<int> pt1, vector<int> pt2) {
    return(abs(pt1[0]-pt2[0]) + abs(pt1[1]-pt2[1]));
}


/////////////////////////////////////////////////////////////////////////////////
//                          Function Definition
//   Name  - Cost
//   Function  - Returns cost of taking a step 
//   Arguments  - vector<int> pt1, vector<int> pt2
/////////////////////////////////////////////////////////////////////////////////

double Cost(vector<int> pt1, vector<int> pt2) {
    vector<vector<int>> gline;
    double cost = 2;
    gline = get_line(pt1[0],pt1[1],pt2[0],pt2[1]);
    for(int i = 0;i<gline.size();i++){
        if ((int)m[gline[i][0]][gline[i][1]] == nmvp){
            cost+=1000;
        }
        Neighbors(5,gline[i][0],gline[i][1]);
        for(int j = 0;j<nei.size(); j++){
            if ((int)m[nei[j][0]][nei[j][1]] == nmvp){
                cost+=100;
            }
        }
    }
    return cost;
}


/////////////////////////////////////////////////////////////////////////////////
//                          Function Definition
//   Name  - findMin
//   Function  - Returns node with lowest fScore 
//   Arguments  - vector<int> pt1, vector<int> pt2
/////////////////////////////////////////////////////////////////////////////////

vector<int> findMin(map<vector<int>, int> openSet) {
    vector<int> v;
    long int min = 100000;
    map<vector<int>, int>::iterator it;
    for ( it = openSet.begin(); it != openSet.end(); it++ )
    {
        if(it -> second < min){
            min = it -> second;
            v = it -> first;
            //cout <<it -> first[0] << " " << it -> first[1] << " " << it -> second << endl;
        }
    }
    return v;
}

vector<vector<int>> reconstruct_path(map<vector<int>, vector<int> >cameFrom, vector<int>current){
    vector<vector<int>> LineFinal;
    vector<int> temp;
    LineFinal.push_back(current);
    map<vector<int>, vector<int> >::iterator it;
    it = cameFrom.find(current);
    while((it != cameFrom.end())) {
        //cout << Dist(current, stop) << " " <<current[0] <<" " << current[1] << " " << cameFrom[current][0] << " " << cameFrom[current][1] << endl;
        temp = cameFrom[current];
        //cameFrom.erase(it);
        current = temp;
        LineFinal.push_back(current);
        it = cameFrom.find(current);
    }
    return LineFinal;

}

int main(int argc, char** argv) {
	//cout << "Starting" <<homedir;
   	string tempst = argv[1];
    coordpath += (tempst+".txt");
    outpath += (tempst+".txt");
    mappath += (tempst+".png");
	vector<vector<int>> gline;
	read_coord(x,y);
	stop[0] = (x+1024);
	stop[1] = (y+1024);
	// cout<<"Coordinates Read "<<stop[0] << " " << stop[1] <<" "<< start[0]<<" "<<start[1] <<endl;
	read_map();
	//cout<<"Map Read"<<endl;
	rot90(3);
	//cout<<"Map Orientation set"<<endl;
	ExtendStartStop();
	int ErrorFree = 0;
	int st_line = 0;
	vector<vector<int>> LineFinal;
    vector<int> current;
	clock_t search_start;
    map<vector<int>, vector<int> > cameFrom;    
    map<vector<int>, double> gScore, fScore;
    map<vector<int>, int> openSet, closedSet;
    map<vector<int>, int>::iterator it;
    gScore[start] = 0;
    fScore[start] = Heuristic(start, stop);
    openSet[start] = fScore[start];
    double tentative_gScore;
    search_start = clock();
    //LineFinal.push_back(start);
    while(openSet.size()!=0){
	if(((double)(clock() - search_start)/1000000) > 600){
		return 0;	
	}
        //cout << "OpenSet Size = " << openSet.size() << endl;
        //cout << "ClosedSet Size = " << closedSet.size() << endl;
        current = findMin(openSet);
        // cout << "Current " << current[0] << " " << current[1] << endl;
        m[current[0]][current[1]] = 50;
        // Mat dataMatrix1(2048,2048,CV_8UC1, m);
        // namedWindow( "Display window", CV_WINDOW_NORMAL);
        // resizeWindow("Display Window", 2048,2048);
        // imshow( "Display window", dataMatrix1 );
        // waitKey(10);
        if (current == stop){
            //cout << "Reached STOP";
            LineFinal = reconstruct_path(cameFrom, current);
            break;
        }
        it = openSet.find(current);
        openSet.erase(it);  
        closedSet[current] = 1;
        //cout << "Erasing " << openSet[current] <<" " << closedSet[current]<< endl;
        Neighbors(1, current[0], current[1]);
        //LineFinal.push_back(current);
        for(int i = 0; i< nei.size(); i++){
            it = closedSet.find(nei[i]);
            if (it == closedSet.end()){   
                //cout << "Found " << nei[i][0] << " " << nei[i][1] << endl;           
                it = openSet.find(nei[i]);
                if(it == openSet.end()){
                    openSet[nei[i]] = 100000;
                    gScore[nei[i]] = 100000;
                }
                tentative_gScore = gScore[current]+Cost(current, nei[i]);
                if(tentative_gScore < gScore[nei[i]]){
                    cameFrom[nei[i]] = current;
                    gScore[nei[i]] = tentative_gScore;
                    fScore[nei[i]] = gScore[nei[i]] + Heuristic(nei[i], stop);
                    openSet[nei[i]] = fScore[nei[i]];
                }    
            }
        }
    }
    //LineFinal.push_back(stop);
	clock_t opt_start2 = clock();
	vector<vector<int>>LF1 = LineFinal;
 //    cout<<LF1.size();
	// vector<vector<int>> LF1;
 //    cout << "Starting Optimization" << endl;   
	// cout << "Map - without optimization" << endl;
    // DisplayPath(LF1);
	LF1 = LineOptimization(LF1);
	// DisplayPath(LF1);
	// cout << "Time taken:- " <<(double)(clock() - search_start)/1000000 << endl;

	vector<idmap> angs = AngleDef(LF1);
	vector<vector<double>> Path;
	vector<double> speeds = SpeedList(LF1,angs);
    ofstream myfile;
    myfile.open(outpath.c_str());
	for (int i = 0; i<speeds.size(); i++) {
        Path.push_back({((LF1[i][0] - 1024.0)/scale) ,((LF1[i][1] - 1024.0)/scale)});
       	//cout << speeds[i] << " " << Path[i][0] << " " << Path[i][1] << endl;
       	myfile << Path[i][0] << "\n" << Path[i][1] << "\n" << speeds[i] << "\n";
    }
    myfile.close();
    cout<<LF1.size();
	return LF1.size();

}


