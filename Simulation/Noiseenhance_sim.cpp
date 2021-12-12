# include <cmath>
# include <cstdlib>
# include <cstring>
# include <ctime>
# include <fstream>
# include <iomanip>
# include <iostream>
# include <random>
# include <sstream>
# include <stdlib.h>
# include <chrono>
# include <vector>

using namespace std;


void Gillespies(int N, int X, int Y, int Z, double L1, double L2, double L3, double B1, double B2, double B3, double n1, double K1, double n2, double K2, double n3, double K3, string header)
{
	string filename;


	// random number generator with time seed
	std::mt19937_64 rng;
    	uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    	std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    	rng.seed(ss);
    	std::uniform_real_distribution<double> unif(0, 1);


	int species[3] = {X,Y,Z};
	double rates[6];
	double R_T;
	double u_t;
	double u_r;
	double t;

	vector<double> histX (400);
	vector<double> histY (400);
	vector<double> histZ (1000);
	

	//vector<vector<double>> histXY (400,vector<double> (400));
	vector<vector<double>> histYZ (400,vector<double> (1000));
	//vector<vector<double>> histXZ (400,vector<double> (400));
	
	
	int N1=0, N2=0, N3=0, N4=0, N5=0, N6=0;

	//start Simulation
	do
	{
		// set reaction rates
		// cout << species[0] << species[1] << species[2] << "\n";
		rates[0] = L1;
		rates[1] = B1*species[0];
		rates[2] = L2*pow(species[2],n2)/(pow(species[2],n2) + pow(K2,n2)) + K1*species[0];
		rates[3] = B2*species[1];
		rates[4] = L3*pow(species[1],n3)/(pow(species[1],n3) + pow(K3,n3));
		rates[5] = B3*species[2];
		
		

		R_T = rates[0] + rates[1] + rates[2] + rates[3] + rates[4] + rates[5];

		// get time after reaction occurs
		u_t = unif(rng);
		t = -log(u_t)/R_T;

		histX[species[0]] = histX[species[0]] + t;
		histY[species[1]] = histY[species[1]] + t;
 		histZ[species[2]] = histZ[species[2]] + t;

		//histXY[species[0]][species[1]] = histXY[species[0]][species[1]] + t;
		histYZ[species[1]][species[2]] = histYZ[species[1]][species[2]] + t;
		//histXZ[species[0]][species[2]] = histXZ[species[0]][species[2]] + t;
		//data << species[0] << "\t" << species[1] << "\t" << t << "\n";

		// Reaction
		u_r = unif(rng);
		if (u_r < rates[0]/R_T)
		{
			species[0] = species[0] + 1;
			N1 = N1 + 1;
		}
		else if (u_r < (rates[0]+rates[1])/R_T)
		{
			species[0] = species[0] - 1;
			N2 = N2 + 1;
		}
		else if (u_r < (rates[0]+rates[1]+rates[2])/R_T)
		{
			species[1] = species[1] + 1;
			N3 = N3 + 1;
		}
		else if (u_r < (rates[0]+rates[1]+rates[2]+rates[3])/R_T)
                {
                        species[1] = species[1] - 1;
			N4 = N4 + 1;
                }
                else if (u_r < (rates[0]+rates[1]+rates[2]+rates[3]+rates[4])/R_T)
                {
                        species[2] = species[2] + 1;
			N5 = N5 + 1;
                }
		else
		{
			species[2] = species[2] - 1;
			N6 = N6 + 1;
		}
	} while (N1<N && N2<N && N3<N && N4<N && N5<N && N6<N);


        //save Data
	int xcut,ycut,zcut;

	for (int ix = X;ix<histX.size(); ix++)
	{
		if (histX[ix++] == 0.)
		{
			xcut = ix;
			break;
		}
	}
	histX.resize(xcut-2);

	ofstream outFileX(header + "_A_file.txt");
   	for (const auto &e : histX) outFileX << e << "\n";


	for (int iy = Y;iy<histY.size(); iy++)
	{
		if (histY[iy++] == 0.)
		{
			ycut = iy;
			break;
		}
	}
	histY.resize(ycut-2);

	ofstream outFileY(header + "_B_file.txt");
   	for (const auto &e : histY) outFileY << e << "\n";

	for (int iz = Z;iz<histZ.size(); iz++)
        {
                if (histZ[iz++] == 0.)
                {
                        zcut = iz;
                        break;
                }
        }
        histZ.resize(zcut-2);

        ofstream outFileZ(header + "_C_file.txt");
        for (const auto &e : histZ) outFileZ << e << "\n";

	
	//histXY.resize(xcut-2);
	//for (int i = 0; i < xcut; i++){
	//	histXY[i].resize(ycut-2);
	//}
	//ofstream outFileXY(header + "_AB_file.txt");
   	//for (const auto &e : histXY){
	//	for (const auto &f : e){
	//		outFileXY << f << "\t";
	//	}
	//	outFileXY << "\n";
	//}
	

	histYZ.resize(ycut-2);
	for (int i = 0; i < ycut; i++){
                histYZ[i].resize(zcut-2);
        }
        ofstream outFileYZ(header + "_BC_file.txt");
	for (const auto &e : histYZ){
	        for (const auto &f : e){
	                outFileYZ << f << "\t";
		}
		outFileYZ << "\n";
	}
	
	
	//histXZ.resize(xcut-2);
	//for (int i = 0; i < xcut; i++){
	//	histXZ[i].resize(zcut-2);
        //}
        //ofstream outFileXZ(header + "_AC_file.txt");
	//for (const auto &e : histXZ){
	//        for (const auto &f : e){
        //               outFileXZ << f << "\t";
        //        }
        //        outFileXZ << "\n";
        //}


	
	ofstream paramfile(header + "_param.txt");
	paramfile << "N" << "\t" << N << "\n";
	paramfile << "A" << "\t" << X << "\n";
	paramfile << "B" << "\t" << Y << "\n";
	paramfile << "C" << "\t" << Z << "\n";
	paramfile << "L1" << "\t" << L1 << "\n";
	paramfile << "L2" << "\t" << L2 << "\n";
	paramfile << "L3" << "\t" << L3 << "\n";
	paramfile << "B1" << "\t" << B1 << "\n";
	paramfile << "B2" << "\t" << B2 << "\n";
	paramfile << "B3" << "\t" << B3 << "\n";
	paramfile << "n1" << "\t" << n1 << "\n";
	paramfile << "n2" << "\t" << n2 << "\n";
	paramfile << "n3" << "\t" << n3 << "\n";
	paramfile << "K1" << "\t" << K1 << "\n";
	paramfile << "K2" << "\t" << K2 << "\n";
        paramfile << "K3" << "\t" << K3 << "\n";
	paramfile.close();

}



int main(int argc, char *argv[])
{

	double N = atoi(argv[1]); //Minimum number for each reaction to occur
	int X = atoi(argv[2]); //Starting conditions
	int Y = atoi(argv[3]);
	int Z = atoi(argv[4]);
	double L1 = atof(argv[5]); //Parameters
	double L2 = atof(argv[6]);
	double L3 = atof(argv[7]);
	double B1 = atof(argv[8]);
	double B2 = atof(argv[9]);
	double B3 = atof(argv[10]);
	double n1 = atof(argv[11]);
	double K1 = atof(argv[12]);
	double n2 = atof(argv[13]);
	double K2 = atof(argv[14]);
	double n3 = atof(argv[15]);
	double K3 = atof(argv[16]);
	string header(argv[17]); //Output Filename

	Gillespies(N,X,Y,Z,L1,L2,L3,B1,B2,B3,n1,K1,n2,K2,n3,K3,header);
}








