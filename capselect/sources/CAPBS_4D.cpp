#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <conio.h>
#include <vector>
#include <functional>
#include <thread>


static float score_fin[50000];
float ff_fin[50000][15];
static int num1_fin[50000];
static int i_count_l;
float min1_l_fin[50000][15];

std::string insert(int i) {
	int j;
	int k;
	std::string printstr;
	printstr = "> <CapScore>\n"
		+ std::to_string(score_fin[i]) + "\n"
		+ "\n"
		+ "> <Spheres>\n"
		+ std::to_string(num1_fin[i]) + "\n"
		+ "\n"
		+ "> <Max(min)>\n";

	for (j = 0; j < num1_fin[i]; j++)
	{
		if (j < num1_fin[i] - 1)
		{

			printstr = printstr + std::to_string(min1_l_fin[i][j]) + ", ";
		}
		else
		{
			printstr = printstr + std::to_string(min1_l_fin[i][j]) + "\n";
		}
		//
	}
	printstr = printstr + "\n" + "> <Distance>\n";
	for (k = 0; k < num1_fin[i]; k++)
	{
		if (k < num1_fin[i] - 1)
		{

			printstr = printstr + std::to_string(ff_fin[i][k]) + ", ";
		}
		else
		{
			printstr = printstr + std::to_string(ff_fin[i][k]) + "\n";
		}
		//
	}
	return printstr;
};


using namespace std;
void read_sdf_frag(string filename, float**& x_l, float**& y_l, float**& z_l, char**& elm_l, char**& lab_l, int*& i_res_l, int*& RecConf)
{
	printf("Reading fragments.sdf");

	string line;

	i_res_l = new int[50000];
	x_l = new float* [50000];
	y_l = new float* [50000];
	z_l = new float* [50000];
	elm_l = new char* [50000];
	lab_l = new char* [50000];
	RecConf = new int[50000];
	ifstream myfile(filename);
	if (myfile.is_open())
	{
		int groupId = 0;

		int i;

		while (!myfile.eof())
		{
			getline(myfile, line);
			if (line.empty()) continue;

			//skip 2 lines
			getline(myfile, line);
			getline(myfile, line);

			getline(myfile, line);
			i_res_l[groupId] = stoi(line.substr(0, 3));

			x_l[groupId] = new float[i_res_l[groupId]];
			y_l[groupId] = new float[i_res_l[groupId]];
			z_l[groupId] = new float[i_res_l[groupId]];
			elm_l[groupId] = new char[i_res_l[groupId]];
			lab_l[groupId] = new char[i_res_l[groupId]];

			for (i = 0; i < i_res_l[groupId]; i++)
			{
				getline(myfile, line);
				x_l[groupId][i] = stof(line.substr(0, 10));
				y_l[groupId][i] = stof(line.substr(10, 10));
				z_l[groupId][i] = stof(line.substr(20, 10));
				elm_l[groupId][i] = line[31];
				lab_l[groupId][i] = line[35];
			}

			
			//RecConf[groupId] = 1;
			groupId++;
			i_count_l = groupId;
		   
			while (getline(myfile, line))
			{
				
				if (line.substr(0, 11) == "> <RecConf>")
				{
					getline(myfile, line);
					RecConf[groupId-1] = stoi(line.substr(0, 1));
				}

				if (line.substr(0, 4) == "$$$$")
					break;
			}
		}

	}
}


void read_sdf_protein(string filename, float**& x_p, float**& y_p, float**& z_p, char**& elm_p, char**& lab_p, int*& i_res_p)
{
	printf("\nLoading %s", filename.c_str());

	string line;

	//float(*i_res_p) = NULL;
	i_res_p = new int[10];

	x_p = new float* [10];
	y_p = new float* [10];
	z_p = new float* [10];
	elm_p = new char* [10];
	lab_p = new char* [10];

	ifstream myfile(filename);
	if (myfile.is_open())
	{
		int proteinId = 0;
        int i_count_p;

		int i;

		while (!myfile.eof())
		{
			getline(myfile, line);
			if (line.empty()) continue;

			//skip 2 lines
			getline(myfile, line);
			getline(myfile, line);

			getline(myfile, line);
			i_res_p[proteinId] = stoi(line.substr(0, 4));

			x_p[proteinId] = new float[i_res_p[proteinId]];
			y_p[proteinId] = new float[i_res_p[proteinId]];
			z_p[proteinId] = new float[i_res_p[proteinId]];
			elm_p[proteinId] = new char[i_res_p[proteinId]];
			lab_p[proteinId] = new char[i_res_p[proteinId]];

			for (i = 0; i < i_res_p[proteinId]; i++)
			{
				getline(myfile, line);
				x_p[proteinId][i] = stof(line.substr(0, 10));
				y_p[proteinId][i] = stof(line.substr(10, 10));
				z_p[proteinId][i] = stof(line.substr(20, 10));
				elm_p[proteinId][i] = line[31];
				lab_p[proteinId][i] = line[35];
			}
			proteinId++;
			i_count_p = proteinId + 1;
			while (getline(myfile, line))
			{
				if (line.substr(0, 4) == "$$$$")
					break;
			}
			break;
		}	
	}
}



bool iterate(std::string input, std::string output, std::string delimiter)
{
	std::ifstream in(input.c_str());
	std::ofstream out(output);
	int it;
	it = 0;
	if (!in)
	{
		std::cerr << "Cannot open the File : " << input << std::endl;
		return false;
	}
	std::string line;
	while (std::getline(in, line))
	{
		if (line.substr(0, 4) == delimiter)
		{
			//printf("$$$$ found - i=%d\n", it);
			out << insert(it) << std::endl;
			out << delimiter << std::endl;
			it++;
		}
		else
		{
			out << line << std::endl;
		}
	}
	in.close();
	out.close();
	return true;
}

bool filecheck(const char* filename)
{
	ifstream file(filename);
	if (!file)
	{
		return false;
	}
	else
	{
		return true;
	}
}

int main()
{
	static int p_check, max_maxl, max_maxll, max_l, max_ll, num_current_1, k1, k2, total, ip2_1, second_l, num_lab, ip4, j2, ip3, num, i, i1, i2, i3, n, n1, nb, i_count1, j1, ii, i4, i5, i6, ip1, ip2, i7, ll, ll2, ll1, ir, ir_p, i20, in, i100, ipp;
	static int  j, ii3, ii4, ii5, ip, counter, k;
	static int  num_lab_l[50000], num_current;
	static float  m, l_check, max_penalty_max1_l, max_penalty_s, sum_x2, y_av, sum_x, sum_y, x_av, x2_av, x_sigma, y_sigma, x_in, y_in, z_in, tt[20], ttt[20], r0, r, r1, r3, rr, rrr, rrrr, min11, a_c_f[500], a_c_t[500], a_s_f[500], a_s_t[500];
	static float  min, r2, r4;
	static char  c[20];
	static int cc, max_label[50000];
	static float  pi, xl, yl, zl, xl0, yl0, zl0;
	static float  x, y, z, rl, x_res, y_res, z_res, x2, y2, z2;
	static float  min1, fi, teta;
	static float  xl_1, yl_1, zl_1, xl_2, yl_2, zl_2, xl_3, yl_3, zl_3, xl_4, yl_4, zl_4;
	static int  num_lab_1, num_lab_2, num_lab_3, num_lab_4;
	float** x_l, ** y_l, ** z_l, ** x_p, ** y_p, ** z_p;
	char** elm_l, ** lab_l, ** elm_p, ** lab_p;
	int* i_res_l, * RecConf, * i_res_p;

	float(*penalty_s)[15];
	penalty_s = new float[50000][15];

	float(*penalty_max1_l)[15];
	penalty_max1_l = new float[50000][15];

	float(*ff)[5][15];
	ff = new float[50000][5][15];

	float(*min1_l)[5][15];
	min1_l = new float[50000][5][15];

	float(*score)[15];
	score = new float[50000][15];

	float(*xll)[15];
	xll = new float[50000][15];

	float(*yll)[15];
	yll = new float[50000][15];

	float(*zll)[15];
	zll = new float[50000][15];

	int(*num1)[15];
	num1 = new int[50000][15];

	float(*x_resl)[5][15];
	x_resl = new float[50000][5][15];

	float(*y_resl)[5][15];
	y_resl = new float[50000][5][15];

	float(*z_resl)[5][15];
	z_resl = new float[50000][5][15];

	printf("    _             __                       \n");
	printf("   /    _.  ._   (_    _   |   _    _  _|_ \n");
	printf("   \\_  (_|  |_)  __)  (/_  |  (/_  (_   |_ \n");
	printf("            |                              \n\n");

	read_sdf_frag("fragments.sdf", x_l, y_l, z_l, elm_l, lab_l, i_res_l, RecConf);
	printf("\nN of FRAGMENTS \t %d", i_count_l);
	printf("\n======================\n");

	pi = 3.14159264;
	for (i4 = 0; i4 < 72; i4++)	//azimut
	{
		teta = i4 * 2. * 5. * pi / 360.;
		a_c_t[i4] = cos(teta);
		a_s_t[i4] = sin(teta);
		//	printf("\n!!!!!!!!!!! i4= %4d    teta= %4.16f   a_c_t= %4.4f    a_s_t= %4.4f ",i4,teta,a_c_t[i4],a_s_t[i4]);
		// _getch();
	}
	for (i1 = 0; i1 < 36; i1++)	//longitude
	{
		fi = -pi / 2. + i1 * 5. * pi / 180.;
		a_c_f[i1] = cos(fi);
		a_s_f[i1] = sin(fi);
		// printf("\n!!!!!!!!!!! fi= %4.4f   a_c_f= %4.4f    a_s_f= %4.4f ",fi,a_c_f[i1],a_s_f[i1]);
		// _getch();
	}

	in = 0;
	k = 0;
	total = 0;
	for (i = 0; i < i_count_l; i++)	//ligand quantity___________________________________
	{
		//_getch();
		if (RecConf[i] == 1 && filecheck("protein1.sdf") == true)
		{
			read_sdf_protein("protein1.sdf", x_p, y_p, z_p, elm_p, lab_p, i_res_p);
		}

		else if (RecConf[i] == 2 && filecheck("protein2.sdf") == true)
		{
			read_sdf_protein("protein2.sdf", x_p, y_p, z_p, elm_p, lab_p, i_res_p);
		}

		else if (filecheck("protein1.sdf") == false)
		{
			printf("\nERROR: No protein file for <RecCon> 1 found!");
			break;
		}

		else if (filecheck("protein2.sdf") == false)
		{
			printf("\nERROR: No protein file for <RecCon> 2 found!");
			break;
		}

		else {
			printf("\nERROR: <RecConf> information invalid.");
			break;
		}


		//i = i + 5;
		//printf("\n");
		num = 0;
		ipp = 0;
		xl_1 = 0.;
		yl_1 = 0.;
		zl_1 = 0.;

		xl_2 = 0.;
		yl_2 = 0.;
		zl_2 = 0.;

		xl_3 = 0.;
		yl_3 = 0.;
		zl_3 = 0.;

		xl_4 = 0.;
		yl_4 = 0.;
		zl_4 = 0.;
		num_lab_1 = 0;
		num_lab_2 = 0;
		num_lab_3 = 0;
		num_lab_4 = 0;
		num_current_1 = 0;
		max_label[i] = 0;
		num_lab_l[i] = 0;

		for (i1 = 0; i1 < i_res_l[i]; i1++)	//ligand quantity
		{
			if (num_lab_1 == 5)
			{
				if (lab_l[i][i1] == 51)
				{
					xl_2 = xl_2 + x_l[i][i1];
					yl_2 = yl_2 + y_l[i][i1];
					zl_2 = zl_2 + z_l[i][i1];
					num_lab_2 = num_lab_2 + 1;
				}
			}
			if (num_lab_1 < 5)
			{
				if (lab_l[i][i1] == 51)
				{
					xl_1 = xl_1 + x_l[i][i1];
					yl_1 = yl_1 + y_l[i][i1];
					zl_1 = zl_1 + z_l[i][i1];
					num_lab_1 = num_lab_1 + 1;
				}
			}
		}
		printf("\n AROMATIC_CAP!!!!!!!!!!! num_lab_1= %d  num_lab_2= %d ", num_lab_1, num_lab_2);
		//_getch();
		for (i1 = 0; i1 < i_res_l[i]; i1++)	//ligand quantity
		{
			if (num_lab_3 == 1)
			{
				if (lab_l[i][i1] == 49)
				{
					xl_4 = x_l[i][i1];
					yl_4 = y_l[i][i1];
					zl_4 = z_l[i][i1];
					num_lab_4 = num_lab_4 + 1;
				}
			}
			if (num_lab_3 < 1)
			{
				if (lab_l[i][i1] == 49)
				{
					//printf("\n===!!!!!!!!!!! i= %4d i1= %d  %d",i,i1,lab_l[i][i1]);
					//__getch();
					xl_3 = x_l[i][i1];
					yl_3 = y_l[i][i1];
					zl_3 = z_l[i][i1];
					num_lab_3 = num_lab_3 + 1;

				}//if
			}

		}//i1
		printf("\n NON_AROMATIC_CAP!!!!!!!!!!! num_lab_3= %d  num_lab_4= %d", num_lab_3, num_lab_4);
		//_getch();
		//printf("\n NON_AROMATIC_CAP!!!!!!!!!!! num_lab= %d", num_lab);
		//_getch();
		if ((num_lab_1 == 0) && (num_lab_2 == 0) && (num_lab_3 == 0) && (num_lab_4 == 0))
		{
			score_fin[i] = -100.;//added on 09/23/2021
			num1_fin[i] = 1;
			goto pp3;
		}
		if ((num_lab_1 == 5) && (num_lab_2 == 5))
		{
			max_label[i] = 3;
			xll[i][1] = xl_1 / num_lab_1;
			yll[i][1] = yl_1 / num_lab_1;
			zll[i][1] = zl_1 / num_lab_1;

			xll[i][2] = xl_2 / num_lab_2;
			yll[i][2] = yl_2 / num_lab_2;
			zll[i][2] = zl_2 / num_lab_2;
			printf("\n 2_AROMATIC_CAPS!!!!!!!!!!! x1= %f  y1= %f  z1= %f", xll[i][1], yll[i][1], zll[i][1]);
			printf("\n 2_AROMATIC_CAPS!!!!!!!!!!! x2= %f  y2= %f  z2= %f", xll[i][2], yll[i][2], zll[i][2]);
			//_getch();
		}
		if ((num_lab_3 == 1) && (num_lab_4 == 1))
		{
			max_label[i] = 3;
			xll[i][1] = xl_3 / num_lab_3;
			yll[i][1] = yl_3 / num_lab_3;
			zll[i][1] = zl_3 / num_lab_3;

			xll[i][2] = xl_4 / num_lab_4;
			yll[i][2] = yl_4 / num_lab_4;
			zll[i][2] = zl_4 / num_lab_4;
			printf("\n 2_NON_AROMATIC_CAPS!!!!!!!!!!! x1= %f  y1= %f  z1= %f", xll[i][1], yll[i][1], zll[i][1]);
			printf("\n 2_NON_AROMATIC_CAPS!!!!!!!!!!! x2= %f  y2= %f  z2= %f", xll[i][2], yll[i][2], zll[i][2]);
		}

		if ((num_lab_1 == 5) && (num_lab_3 == 1))
		{
			max_label[i] = 3;
			xll[i][1] = xl_1 / num_lab_1;
			yll[i][1] = yl_1 / num_lab_1;
			zll[i][1] = zl_1 / num_lab_1;

			xll[i][2] = xl_3 / num_lab_3;
			yll[i][2] = yl_3 / num_lab_3;
			zll[i][2] = zl_3 / num_lab_3;
			printf("\n AROM_NON_AROM_CAPS!!!!!!!!!!! x1= %f  y1= %f  z1= %f", xll[i][1], yll[i][1], zll[i][1]);
			printf("\n AROM_NON_AROM_CAPS!!!!!!!!!!! x2= %f  y2= %f  z2= %f", xll[i][2], yll[i][2], zll[i][2]);
		}

		if (((num_lab_1 == 5) && (num_lab_2 == 0)) && ((num_lab_3 == 0) && (num_lab_4 == 0)))
		{
			max_label[i] = 2;
			xll[i][1] = xl_1 / num_lab_1;
			yll[i][1] = yl_1 / num_lab_1;
			zll[i][1] = zl_1 / num_lab_1;
			printf("\n SINGLE_CAP!!!!!!!!!!! x1= %f  y1= %f  z1= %f", xll[i][1], yll[i][1], zll[i][1]);
			//_getch();
		}

		if (((num_lab_1 == 0) && (num_lab_2 == 0)) && ((num_lab_3 == 1) && (num_lab_4 == 0)))
		{
			max_label[i] = 2;
			xll[i][1] = xl_3 / num_lab_3;
			yll[i][1] = yl_3 / num_lab_3;
			zll[i][1] = zl_3 / num_lab_3;
			printf("\n SINGLE_CAP!!!!!!!!!!! x1= %f  y1= %f  z1= %f", xll[i][1], yll[i][1], zll[i][1]);
			//_getch();
		}

		num_lab_l[i] = num_lab_1 + num_lab_2 + num_lab_3 + num_lab_4;
		if ((num_lab_l[i] == 1) || (num_lab_l[i] == 5))
		{
			max_label[i] = 2;
			num_current_1 = 1;
		}
		if ((num_lab_l[i] == 10) || (num_lab_l[i] == 6) || (num_lab_l[i] == 2))
		{
			max_label[i] = 3;
			num_current_1 = 1;
		}
		//_getch();
		for (cc = 1; cc < max_label[i]; cc++)
		{
			printf("\nCAP COORDINATES_ALL           frag= %4d    max_label= %4d    xl= %4.4f    yl= %4.4f    zl= %4.4f", i, max_label[i], xll[i][cc], yll[i][cc], zll[i][cc]);
			//_getch();
		}

		for (num_current = 1; num_current < max_label[i]; num_current++)
		{
			num = 0;
			x_in = xll[i][num_current];
			y_in = yll[i][num_current];
			z_in = zll[i][num_current];
			printf("\nCAP COORDINATES rd= % 4d      frag= %4d    max_label= %4d    xl= %4.4f    yl= %4.4f    zl= %4.4f", num_current, i, max_label[i], xll[i][num_current], yll[i][num_current], zll[i][num_current]);
			//_getch();
			//_getch();
			//5. ______________________ XX1,YY1,ZZ1 FORMATION SPHERICAL KOORDINATES_________
			//***************************  1 STEP ******************************************

			pi = 3.14159264;
			r = 3.0; //was 3.0 on 09/17
			min11 = 0.;
			if (num_lab_l[i] == 5) { r = 3.5; }
			for (i4 = 0; i4 < 72; i4++)	//azimut
			{
				//teta=i4*2.*5.*pi/360.;
				for (i1 = 0; i1 < 36; i1++)	//longitude
				{
					//fi=-pi/2.+i1*5.*pi/180.;
				 //x,y,z
					z = r * a_s_f[i1];
					x = r * a_c_f[i1] * a_s_t[i4];
					y = r * a_c_f[i1] * a_c_t[i4];

					//	printf("\nFIRST!!!!!!!!!!! x= %4.4f   y= %4.4f    z= %4.4f ",x,y,z);
					//  _getch();
					//  printf("\nCAP COORDINATES             frag= %4d    max_label= %4d    xl= %4.4f    yl= %4.4f    zl= %4.4f", i, max_label[i], xll[i][num_current], yll[i][num_current], zll[i][num_current]);
					//  _getch();
					//_________ SEARCH of MIN _______


					z = z + zll[i][num_current];
					y = y + yll[i][num_current];
					x = x + xll[i][num_current];
					//	printf("\n!!!!!!!!!!! x= %4.4f   y= %4.4f    z= %4.4f ",x,y,z);
					//  _getch();
					l_check = 1.3;//was 1.3 on 09/17
					if (num_lab_l[i] == 5) { l_check = 1.1; }

					ip = 0;
					for (i3 = 0; i3 < i_res_l[i]; i3++)
					{
						if ((lab_l[i][i3] != 49) && (elm_l[i][i3] != 72) && (lab_l[i][i3] != 51))
						{
							r2 = sqrt((x - x_l[i][i3]) * (x - x_l[i][i3]) + (y - y_l[i][i3]) * (y - y_l[i][i3]) + (z - z_l[i][i3]) * (z - z_l[i][i3]));
							r1 = sqrt((x_in - x_l[i][i3]) * (x_in - x_l[i][i3]) + (y_in - y_l[i][i3]) * (y_in - y_l[i][i3]) + (z_in - z_l[i][i3]) * (z_in - z_l[i][i3]));
							if ((r2 < r) || (r1 < l_check)) { ip = 1; }
						}//if
					}//i3
				 //if(ip==1){goto c_lig1;}

					p_check = 3.0;
					if (num_lab_l[i] == 5) { p_check = 2.0; }

					ip1 = 0;
					for (i3 = 0; i3 < i_res_p[0]; i3++)
					{
						{
							r2 = sqrt((x - x_p[0][i3]) * (x - x_p[0][i3]) + (y - y_p[0][i3]) * (y - y_p[0][i3]) + (z - z_p[0][i3]) * (z - z_p[0][i3]));
							r1 = sqrt((x_in - x_p[0][i3]) * (x_in - x_p[0][i3]) + (y_in - y_p[0][i3]) * (y_in - y_p[0][i3]) + (z_in - z_p[0][i3]) * (z_in - z_p[0][i3]));
						}
						//if ((r2 < p_check) || (r1 < r)) { ip1 = 1; }
						if ((r2 < p_check)) { ip1 = 1; }
					}//i3

					if ((ip != 1) && (ip1 != 1)) { ipp = 1; }
					if ((ip == 1) || (ip1 == 1)) { goto c_lig1; }

					min = 1000.;
					for (i2 = 0; i2 < i_res_p[0]; i2++)	//
					{
						{
							r1 = sqrt((x - x_p[0][i2]) * (x - x_p[0][i2]) + (y - y_p[0][i2]) * (y - y_p[0][i2]) + (z - z_p[0][i2]) * (z - z_p[0][i2]));
						}
						if (min > r1) { min = r1; x2 = x - xl * 0.; y2 = y - yl * 0.; z2 = z - zl * 0.; }
					}//i2
					if (min11 < min) { min11 = min; x_res = x2; y_res = y2; z_res = z2; }
				c_lig1:;
				}//i1
			}//i4

			if (ipp == 0)
			{
				num1[i][num_current] = 0;
				min1_l[i][num_current][0] = 0.;
				ff[i][num_current][0] = 0.;
				//fprintf(stream_w, "\n%4d    %4d %4d     %4.4f     %4.4f     %4.4f     %4.4f    %4.4f    %4.4f    %4.4f  \n", i + 1, num, 0, 0., 0., 0., 0., xl, yl, zl);

				//fprintf(stream_s, "\n");
				if (num_lab_l[i] == 2)
				{
					if (num_current == 2)
					{
						goto pp;
					}
					if (num_current == 1)
					{
						//num_current_1 = num_current_1 + 1;
						goto pp1;
					}
				}
				goto pp;
			}

			x_resl[i][num_current][0] = x_res;
			y_resl[i][num_current][0] = y_res;
			z_resl[i][num_current][0] = z_res;
			min1_l[i][num_current][0] = min11;

			ff[i][num_current][0] = sqrt((x_in - x_resl[i][num_current][0]) * (x_in - x_resl[i][num_current][0]) + (y_in - y_resl[i][num_current][0]) * (y_in - y_resl[i][num_current][0]) + (z_in - z_resl[i][num_current][0]) * (z_in - z_resl[i][num_current][0]));
			printf("\n CAP                           frag= %4d    num= %4d    MinMax= %4.4f    ff= %4.4f    x_res= %4.4f    y_res= %4.4f    z_res= %4.4f", i, num, min1_l[i][num_current][0], ff[i][num_current][0], x_resl[i][num_current][0], y_resl[i][num_current][0], z_resl[i][num_current][0]);
			//_getch();
			//fprintf(stream_w,"\n %4d    %f     %f     %f     %f",i,min11,x_res, y_res, z_res);
			//printf("\nCAP COORDINATES@@@@@@@             frag= %4d    max_label= %4d    xl= %4.4f    yl= %4.4f    zl= %4.4f", i, max_label[i], xll[i][num_current], yll[i][num_current], zll[i][num_current]);
			//_getch();
			counter = counter + ipp;
			//goto pp;

			//***************************  2 STEP ******************************************
			num = 1;


		www:
			ipp = 0;
			rrr = 3.;//was 3.0 before 09/17
			rrrr = 2.;
			zl = z_res;
			yl = y_res;
			xl = x_res;


			min11 = 0.;
			for (i4 = 0; i4 < 72; i4++)	//azimut
			{
				//teta=i4*2.*5.*pi/360.;
				for (i1 = 0; i1 < 36; i1++)	//longitude
				{
					//fi=-pi/2.+i1*5.*pi/180.;
				 //x,y,z
					z = rrrr * a_s_f[i1];
					x = rrrr * a_c_f[i1] * a_s_t[i4];
					y = rrrr * a_c_f[i1] * a_c_t[i4];
					//printf("\n!!!!!!!!!!! i= %4d  i1= %d x= %f y= %f z= %f",i,i1,x,y,z);
					//_getch();

			//_________ SEARCH of MIN _______

					z = z + zl;
					y = y + yl;
					x = x + xl;

					ip = 0;
					l_check = 1.3;
					if (num_lab_l[i] == 5) { l_check = 1.1; }

					for (i3 = 0; i3 < i_res_l[i]; i3++)
					{
						if ((lab_l[i][i3] != 49) && (elm_l[i][i3] != 72) && (lab_l[i][i3] != 51))
						{
							r2 = sqrt((x - x_l[i][i3]) * (x - x_l[i][i3]) + (y - y_l[i][i3]) * (y - y_l[i][i3]) + (z - z_l[i][i3]) * (z - z_l[i][i3]));
							r1 = sqrt((xl - x_l[i][i3]) * (xl - x_l[i][i3]) + (yl - y_l[i][i3]) * (yl - y_l[i][i3]) + (zl - z_l[i][i3]) * (zl - z_l[i][i3]));
							if ((r2 < rrr) || (r1 < l_check)) { ip = 1; }

						}//if
					}//i3
					ip1 = 0;
					for (i3 = 0; i3 < i_res_p[0]; i3++)
					{
						{
							r2 = sqrt((x - x_p[0][i3]) * (x - x_p[0][i3]) + (y - y_p[0][i3]) * (y - y_p[0][i3]) + (z - z_p[0][i3]) * (z - z_p[0][i3]));
							r1 = sqrt((xl - x_p[0][i3]) * (xl - x_p[0][i3]) + (yl - y_p[0][i3]) * (yl - y_p[0][i3]) + (zl - z_p[0][i3]) * (zl - z_p[0][i3]));
						}
						if ((r2 < 2.0) || (r1 < 3.0)) { ip1 = 1; }

					}//i3

					ip2 = 0;

					if (num > 1)
					{
						for (i7 = 0; i7 < num; i7++)
						{
							r1 = sqrt((x - x_resl[i][num_current][i7]) * (x - x_resl[i][num_current][i7]) + (y - y_resl[i][num_current][i7]) * (y - y_resl[i][num_current][i7]) + (z - z_resl[i][num_current][i7]) * (z - z_resl[i][num_current][i7]));
							//r2 = sqrt((x - x_in) * (x - x_in) + (y - y_in) * (y - y_in) + (z - z_in) * (z - z_in));//added on 09/17/21
							if (r1 < 2.0) { ip2 = 1; }
							//if ((r1 < 2.0) || (r2 < 3.0)) { ip2 = 1; }//added on 09/17/21
						}//i7
					}
					ip2_1 = 0;
					if (num == 1)
					{
						r2 = sqrt((x - x_in) * (x - x_in) + (y - y_in) * (y - y_in) + (z - z_in) * (z - z_in));//added on 09/17/21
						if (num_lab_l[i] == 5)
						{
							if ((r2 < 3.5)) { ip2_1 = 1; }//added on 09/17/21
							//if ((r2 < 5.33)) { ip2_1 = 1; }//added on 09/18/21
						}
						if ((num_lab_l[i] == 1) || (num_lab_l[i] == 2))
						{
							if ((r2 < 3.0)) { ip2_1 = 1; }//added on 09/17/21
							//if ((r2 < 4.84)) { ip2_1 = 1; }//added on 09/18/21
						}
					}




					ip3 = 0;
					if (num > 1)
					{
						for (j2 = (num - 2); j2 < (num - 1); j2++)
						{
							r2 = sqrt((x - x_resl[i][num_current][j2]) * (x - x_resl[i][num_current][j2]) + (y - y_resl[i][num_current][j2]) * (y - y_resl[i][num_current][j2]) + (z - z_resl[i][num_current][j2]) * (z - z_resl[i][num_current][j2]));
							//if (r2 < 2.82) { ip3 = 1; } //90degree
							if (r2 < 3.46) { ip3 = 1; } //60degree
						}
					}
					ip4 = 0;

					if ((ip != 1) && (ip1 != 1) && (ip2 != 1) && (ip3 != 1) && (ip4 != 1) && (ip2_1 != 1)) { ipp = 1; }
					if ((ip == 1) || (ip1 == 1) || (ip2 == 1) || (ip3 == 1) || (ip4 == 1) || (ip2_1 == 1)) { goto c_lig; }

					min = 1000.;
					for (i2 = 0; i2 < i_res_p[0]; i2++)	//
					{
						//if (elm_p[0][i2] != 72)
						{
							r1 = sqrt((x - x_p[0][i2]) * (x - x_p[0][i2]) + (y - y_p[0][i2]) * (y - y_p[0][i2]) + (z - z_p[0][i2]) * (z - z_p[0][i2]));
						}
						if (min > r1) { min = r1; x2 = x - xl * 0.; y2 = y - yl * 0.; z2 = z - zl * 0.; }

					}//i2
					if (min11 < min) { min11 = min; x_res = x2; y_res = y2; z_res = z2; }

				c_lig:;
				}//i1
			}//i4
			if ((ipp == 0) && (num == 1))
				//if (ipp==0)
			{
				num1[i][num_current] = 1;
				ff[i][num_current][0] = 0.;
				min1_l[i][num_current][num] = min1_l[i][num_current][0];
				//score[i] = 0;
				//fprintf(stream_w, "\n%4d    %4d %4d     %4.4f     %4.4f     %4.4f     %4.4f     %4.4f     %4.4f     %4.4f  \n", i + 1, num, 1, min1_l[i][0], 0., 0., min1_l[i][0], x_resl[i][0], y_resl[i][0], z_resl[i][0]);
				if ((num_lab_l[i] == 10) || (num_lab_l[i] == 6) || (num_lab_l[i] == 2))
				{
					if (num_current == 2)
					{

						goto pp;
					}
					if (num_current == 1)
					{

						//	num_current_1 = num_current_1 + 1;
						goto pp1;
					}
				}
				goto pp;
			}



			if ((ipp == 0) && (num > 1) && (num <= 9))
			{
				num1[i][num_current] = num;
				//fprintf(stream_w, "\n%4d    %4d", i + 1, num);
				//fprintf(stream_s, "\n");

				if ((num_lab_l[i] == 10) || (num_lab_l[i] == 6) || (num_lab_l[i] == 2))
				{
					if (num_current == 2)
					{

						goto pp;
					}

					if (num_current == 1)
					{

						//	num_current_1 = num_current_1 + 1;
						goto pp1;
					}
				}
				goto pp;

			}
			x_resl[i][num_current][num] = x_res;
			y_resl[i][num_current][num] = y_res;
			z_resl[i][num_current][num] = z_res;
			min1_l[i][num_current][num] = min11;

			ff[i][num_current][num] = sqrt((x_resl[i][num_current][num] - x_in) * (x_resl[i][num_current][num] - x_in) + (y_resl[i][num_current][num] - y_in) * (y_resl[i][num_current][num] - y_in) + (z_resl[i][num_current][num] - z_in) * (z_resl[i][num_current][num] - z_in));
			printf("\n                            frag= %4d    num= %4d    MinMax= %f    ff= %f    x_res= %f    y_res= %f    z_res= %f", i, num, min1_l[i][num_current][num], ff[i][num_current][num], x_resl[i][num_current][num], y_resl[i][num_current][num], z_resl[i][num_current][num]);
			//__getch();
			if (num > 8)
			{
				num1[i][num_current] = num + 1;
				//fprintf(stream_w, "\n%4d    %4d", i + 1, num);
				for (i5 = 0; i5 < num + 1; i5++)
				{
					if (i5 == (num - 1))
					{
						//fprintf(stream_w, "\n              %4d      %4.4f     %4.4f     %4.4f      %4.4f     %4.4f     %4.4f \n", i5 + 1, min1_l[i][i5][num_current], min1_l[i][i5][num_current], min1_l[i][i5][num_current], x_resl[i][i5][num_current], y_resl[i][i5][num_current], z_resl[i][i5][num_current]);
						goto ww3;
					}
					//fprintf(stream_w, "\n              %4d     %4.4f     %4.4f    %4.4f      %4.4f     %4.4f     %4.4f ", i5 + 1, min1_l[i][i5][num_current], min1_l[i][i5][num_current], min1_l[i][i5][num_current], x_resl[i][i5][num_current], y_resl[i][i5][num_current], z_resl[i][i5][num_current]);
				ww3:;
				}
				//fprintf(stream_s, "\n");
				if ((num_lab_l[i] == 10) || (num_lab_l[i] == 6) || (num_lab_l[i] == 2))
				{
					if (num_current == 2)
					{

						goto pp;
					}
					if (num_current == 1)
					{

						goto pp1;
					}
				}
				goto pp;

			}//if
			num = num + 1;
			goto www;

		pp1:;
			//printf("\n2********* i= %d min1= %f", i, min1);
			//      _getch();
		}

	pp:
		//printf("\nCOUNTER i= %4d    counter= %d", i, counter);
	//	}//i
	//______________________score_penalty____________________________________

		if ((num_lab_l[i] == 1) || (num_lab_l[i] == 5))
		{
			for (j1 = 1; j1 < 2; j1++)
			{
				if (num1[i][j1] <= 5)
				{
					penalty_s[i][j1] = (5 - num1[i][j1]) * (5 - num1[i][j1]);
				}
				else
				{
					penalty_s[i][j1] = 0.;
				}
			}
		}
		if ((num_lab_l[i] == 10) || (num_lab_l[i] == 6) || (num_lab_l[i] == 2))
		{
			for (j2 = 1; j2 < 3; j2++)
			{
				if (num1[i][j2] <= 5)
				{
					penalty_s[i][j2] = (5 - num1[i][j2]) * (5 - num1[i][j2]);
				}
				else
				{
					penalty_s[i][j2] = 0.;
				}
			}
		}


		if ((num_lab_l[i] == 1) || (num_lab_l[i] == 5))
		{
			for (j1 = 1; j1 < 2; j1++)
			{
				penalty_s[i][j1] = (penalty_s[i][j1] * 10.) / 25.;
			}
		}
		if ((num_lab_l[i] == 10) || (num_lab_l[i] == 6) || (num_lab_l[i] == 2))
		{
			for (j2 = 1; j2 < 3; j2++)
			{
				penalty_s[i][j2] = (penalty_s[i][j2] * 10.) / 25.;

			}
		}


		if ((num_lab_l[i] == 1) || (num_lab_l[i] == 5))
		{
			for (j1 = 1; j1 < 2; j1++)
			{
				if (num1[i][j1] > 9)
				{
					if ((min1_l[i][j1][9] >= 7.) && (min1_l[i][j1][9] <= 20.))
					{
						penalty_max1_l[i][j1] = pow((7.0 - min1_l[i][j1][9]), 2.);
					}

					if (min1_l[i][j1][9] > 20.)
					{
						penalty_max1_l[i][j1] = 169.;
					}

					if (min1_l[i][j1][9] < 7.)
					{
						penalty_max1_l[i][j1] = 0.;
					}

				}
				if (num1[i][j1] <= 9)
				{
					penalty_max1_l[i][j1] = 0.;
				}
			}
		}
		if ((num_lab_l[i] == 10) || (num_lab_l[i] == 6) || (num_lab_l[i] == 2))
		{
			for (j2 = 1; j2 < 3; j2++)
			{
				if (num1[i][j2] > 9)
				{
					if ((min1_l[i][j2][9] >= 7.) && (min1_l[i][j2][9] <= 20.))
					{
						penalty_max1_l[i][j2] = pow((7.0 - min1_l[i][j2][9]), 2.);
					}

					if (min1_l[i][j2][9] > 20.)
					{
						penalty_max1_l[i][j2] = 169.;
					}

					if (min1_l[i][j2][9] < 7.)
					{
						penalty_max1_l[i][j2] = 0.;
					}

				}
				if (num1[i][j2] <= 9)
				{
					penalty_max1_l[i][j2] = 0.;
				}
			}
		}




		if ((num_lab_l[i] == 1) || (num_lab_l[i] == 5))
		{
			for (j1 = 1; j1 < 2; j1++)
			{
				penalty_max1_l[i][j1] = (penalty_max1_l[i][j1] * 10.) / 169.;
			}
		}
		if ((num_lab_l[i] == 10) || (num_lab_l[i] == 6) || (num_lab_l[i] == 2))
		{
			for (j2 = 1; j2 < 3; j2++)
			{
				penalty_max1_l[i][j2] = (penalty_max1_l[i][j2] * 10.) / 169.;
			}
		}



		if ((num_lab_l[i] == 1) || (num_lab_l[i] == 5))
		{
			score[i][1] = 10. - penalty_s[i][1] - penalty_max1_l[i][1];
			score_fin[i] = score[i][1];
			num1_fin[i] = num1[i][1];
			for (k = 0; k < num1[i][1]; k++)
			{
				min1_l_fin[i][k] = min1_l[i][1][k];
				ff_fin[i][k] = ff[i][1][k];
				//	printf("\nSSSSSSS                       frag= %4d     MinMax= %4.4f    ff= %4.4f   ", i, min1_l[i][1][k], ff[i][1][k]);
				//	_getch();
			}

		}

		if ((num_lab_l[i] == 10) || (num_lab_l[i] == 6) || (num_lab_l[i] == 2))
		{
			for (j2 = 1; j2 < 3; j2++)
			{
				score[i][j2] = 10. - penalty_s[i][j2] - penalty_max1_l[i][j2];
				m = score[i][1];
				score_fin[i] = m;
				num1_fin[i] = num1[i][1];
				for (k1 = 0; k1 < num1[i][j2]; k1++)
				{
					min1_l_fin[i][k1] = min1_l[i][1][k1];
					ff_fin[i][k1] = ff[i][1][k1];
				}
				if (m < score[i][j2])
				{
					score_fin[i] = score[i][2];
					num1_fin[i] = num1[i][2];
					for (k2 = 0; k2 < num1[i][2]; k2++)
					{
						min1_l_fin[i][k2] = min1_l[i][2][k2];
						ff_fin[i][k2] = ff[i][2][k2];
					}
				}
			}
		}


		printf("\nCOUNTER i= %4d    counter= %d", i, counter);
	pp3:;
	}

	/*


		for (i = 0; i < i_count_l; i++)//______________________score_penalty____________________________________
		{
			if ((num_lab_l[i] == 1) || (num_lab_l[i] == 5))
			{
				for (j1 = 1; j1 < 2; j1++)
				{
					if (num1[i][j1] <= 5)
					{
						penalty_s[i][j1] = (5 - num1[i][j1]) * (5 - num1[i][j1]);
					}
					else
					{
						penalty_s[i][j1] = 0.;
					}
				}
			}
			if ((num_lab_l[i] == 2))
			{
				for (j2 = 1; j2 < 3; j2++)
				{
					if (num1[i][j2] <= 5)
					{
						penalty_s[i][j2] = (5 - num1[i][j2]) * (5 - num1[i][j2]);
					}
					else
					{
						penalty_s[i][j2] = 0.;
					}
				}
			}

		}

		for (i = 0; i < i_count_l; i++)
		{
			if ((num_lab_l[i] == 1) || (num_lab_l[i] == 5))
			{
				for (j1 = 1; j1 < 2; j1++)
				{
					penalty_s[i][j1] = (penalty_s[i][j1] * 10.) / 25.;
				}
			}
			if ((num_lab_l[i] == 2))
			{
				for (j2 = 1; j2 < 3; j2++)
				{
					penalty_s[i][j2] = (penalty_s[i][j2] * 10.) / 25.;

				}
			}

		}

		for (i = 0; i < i_count_l; i++)//______________________MaxMin_penalty
		{
			if ((num_lab_l[i] == 1) || (num_lab_l[i] == 5))
			{
				for (j1 = 1; j1 < 2; j1++)
				{
					if (num1[i][j1] > 9)
					{
						if ((min1_l[i][j1][9] >= 7.) && (min1_l[i][j1][9] <= 20.))
						{
							penalty_max1_l[i][j1] = pow((7.0 - min1_l[i][j1][9]), 2.);
						}

						if (min1_l[i][j1][9] > 20.)
						{
							penalty_max1_l[i][j1] = 169.;
						}

						if (min1_l[i][j1][9] < 7.)
						{
							penalty_max1_l[i][j1] = 0.;
						}

					}
					if (num1[i][j1] <= 9)
					{
						penalty_max1_l[i][j1] = 0.;
					}
				}
			}
			if ((num_lab_l[i] == 2))
			{
				for (j2 = 1; j2 < 3; j2++)
				{
					if (num1[i][j2] > 9)
					{
						if ((min1_l[i][j2][9] >= 7.) && (min1_l[i][j2][9] <= 20.))
						{
							penalty_max1_l[i][j2] = pow((7.0 - min1_l[i][j2][9]), 2.);
						}

						if (min1_l[i][j2][9] > 20.)
						{
							penalty_max1_l[i][j2] = 169.;
						}

						if (min1_l[i][j2][9] < 7.)
						{
							penalty_max1_l[i][j2] = 0.;
						}

					}
					if (num1[i][j2] <= 9)
					{
						penalty_max1_l[i][j2] = 0.;
					}
				}
			}

		}


		for (i = 0; i < i_count_l; i++)
		{
			if ((num_lab_l[i] == 1) || (num_lab_l[i] == 5))
			{
				for (j1 = 1; j1 < 2; j1++)
				{
					penalty_max1_l[i][j1] = (penalty_max1_l[i][j1] * 10.) / 169.;
				}
			}
			if ((num_lab_l[i] == 2) )
			{
				for (j2 = 1; j2 < 3; j2++)
				{
					penalty_max1_l[i][j2] = (penalty_max1_l[i][j2] * 10.) / 169.;
				}
			}

		}

		for (i = 0; i < i_count_l; i++)
		{
			if ((num_lab_l[i] == 1) || (num_lab_l[i] == 5))
			{
				score[i][1] = 10. - penalty_s[i][1] - penalty_max1_l[i][1];
				score_fin[i] = score[i][1];
				num1_fin[i] = num1[i][1];
				for (k = 0; k < num1[i][1]; k++)
				{
					min1_l_fin[i][k] = min1_l[i][1][k];
					ff_fin[i][k] = ff[i][1][k];
				//	printf("\nSSSSSSS                       frag= %4d     MinMax= %4.4f    ff= %4.4f   ", i, min1_l[i][1][k], ff[i][1][k]);
				//	_getch();
				}

			}

			if ((num_lab_l[i] == 2) )
			{
				for (j2 = 1; j2 < 3; j2++)
				{
					score[i][j2] = 10. - penalty_s[i][j2] - penalty_max1_l[i][j2];
					m = score[i][1];
					score_fin[i] = m;
					num1_fin[i] = num1[i][1];
					for (k1 = 0; k1 < num1[i][j2]; k1++)
					{
						min1_l_fin[i][k1] = min1_l[i][1][k1];
						ff_fin[i][k1] = ff[i][1][k1];
					}
					if (m < score[i][j2])
					{
						score_fin[i] = score[i][2];
						num1_fin[i] = num1[i][2];
						for (k2 = 0; k2 < num1[i][2]; k2++)
						{
							min1_l_fin[i][k2] = min1_l[i][2][k2];
							ff_fin[i][k2] = ff[i][2][k2];
						}
					}
				}
			}

		}

		*/
		//writeout("frags600AT2.sdf", "CapSelect.sdf", "$$$$");

				//std::string input = "frags_600cpla2.sdf";
				//std::string input = "frags_CB2.sdf";
				//std::string input = "cpla2_30000_1.sdf";
	std::string input = "fragments.sdf";
	std::string output = "CapSelect.sdf";
	std::string delimiter = "$$$$";
	bool res = iterate(input, output, delimiter);
	if (res)
	{
		printf("Done! CapSelect.sdf written \n");
	}
	else
	{
		printf("Error writing into the file!");
	};

}
//__getch();
//_getch();


