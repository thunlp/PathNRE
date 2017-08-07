#include <cstring>
#include <cstdlib>
#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <float.h>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;


string version = "";

int output_model = 0;

int num_threads = 10;
int trainTimes = 25;
float alpha = 0.02;
float reduce = 0.98;
int tt,tt1;
int dimensionC = 230;
int dimensionWPE = 5;
int window = 3;
int limit = 30;
float marginPositive = 2.5;
float marginNegative = 0.5;
float margin = 2;
float Belt = 0.001;
float *matrixB1, *matrixRelation, *matrixW1, *matrixRelationDao, *matrixRelationPr, *matrixRelationPrDao;
float *matrixB1_egs, *matrixRelation_egs, *matrixW1_egs, *matrixRelationPr_egs;
float *matrixB1_exs, *matrixRelation_exs, *matrixW1_exs, *matrixRelationPr_exs;
float *wordVecDao,*wordVec_egs,*wordVec_exs;
float *positionVecE1, *positionVecE2, *matrixW1PositionE1, *matrixW1PositionE2;
float *positionVecE1_egs, *positionVecE2_egs, *matrixW1PositionE1_egs, *matrixW1PositionE2_egs, *positionVecE1_exs, *positionVecE2_exs, *matrixW1PositionE1_exs, *matrixW1PositionE2_exs;
float *matrixW1PositionE1Dao;
float *matrixW1PositionE2Dao;
float *positionVecDaoE1;
float *positionVecDaoE2;
float *matrixW1Dao;
float *matrixB1Dao;
double mx = 0;
int batch = 16;
int npoch;
int len;
float rate = 1;
FILE *logg;

float *wordVec;
int wordTotal, dimension, relationTotal;
int PositionMinE1, PositionMaxE1, PositionTotalE1,PositionMinE2, PositionMaxE2, PositionTotalE2;
map<string,int> wordMapping;
vector<string> wordList;
map<string,int> relationMapping;
vector<int *> trainLists, trainPositionE1, trainPositionE2;
vector<int> trainLength;
vector<int> headList, tailList, relationList;
vector<int *> testtrainLists, testPositionE1, testPositionE2;
vector<int> testtrainLength;
vector<int> testheadList, testtailList, testrelationList;
vector<std::string> nam;

map<string,vector<int> > bags_train, bags_test;
map<string,int> bags_train_id,bags_test_id;
map<string,vector<int *> >train_path, test_path;
map<int, vector<int> >train_pair,test_pair, test_train_pair;
map<pair<int,int>, vector<int> >train_pair_bags,test_pair_bags, test_train_pair_bags;

map<int, int> rel_pred;


void init() {
	FILE *f = fopen("../data/vec4.bin", "rb");
	fscanf(f, "%d", &wordTotal);
	fscanf(f, "%d", &dimension);
	cout<<"wordTotal=\t"<<wordTotal<<endl;
	cout<<"Word dimension=\t"<<dimension<<endl;
	PositionMinE1 = 0;
	PositionMaxE1 = 0;
	PositionMinE2 = 0;
	PositionMaxE2 = 0;
	wordVec = (float *)malloc((wordTotal+1) * dimension * sizeof(float));
	wordList.resize(wordTotal+1);
	wordList[0] = "UNK";
	for (int b = 1; b <= wordTotal; b++) {
		string name = "";
		while (1) {
			char ch = fgetc(f);
			if (feof(f) || ch == ' ') break;
			if (ch != '\n') name = name + ch;
		}
		int last = b * dimension;
		float smp = 0;
		for (int a = 0; a < dimension; a++) {
			fread(&wordVec[a + last], sizeof(float), 1, f);
			smp += wordVec[a + last]*wordVec[a + last];
		}
		smp = sqrt(smp);
		for (int a = 0; a< dimension; a++)
			wordVec[a+last] = wordVec[a+last] / smp;
		wordMapping[name] = b;
		wordList[b] = name;
	}
	wordTotal+=1;
	fclose(f);
	char buffer[1000];
	f = fopen("../data/relation2id.txt", "r");
	while (fscanf(f,"%s",buffer)==1) {
		int id;
		fscanf(f,"%d",&id);
		relationMapping[(string)(buffer)] = id;
		relationTotal++;
		nam.push_back((std::string)(buffer));
	}
	fclose(f);
	cout<<"relationTotal:\t"<<relationTotal<<endl;
	
	f = fopen("../data/train.txt", "r");

	while (fscanf(f,"%s",buffer)==1)  {
		string e1 = buffer;
		fscanf(f,"%s",buffer);
		string e2 = buffer;
		fscanf(f,"%s",buffer);
		string head_s = (string)(buffer);
		int head = wordMapping[(string)(buffer)];
		fscanf(f,"%s",buffer);
		int tail = wordMapping[(string)(buffer)];
		string tail_s = (string)(buffer);
		fscanf(f,"%s",buffer);

		bags_train[e1+"\t"+e2+"\t"+(string)(buffer)].push_back(headList.size());

		int num = relationMapping[(string)(buffer)];
		int len = 0, lefnum = 0, rignum = 0;
		std::vector<int> tmpp;
		while (fscanf(f,"%s", buffer)==1) {
			std::string con = buffer;
			if (con=="###END###") break;
			int gg = wordMapping[con];
			if (con == head_s) lefnum = len;
			if (con == tail_s) rignum = len;
			len++;
			tmpp.push_back(gg);
		}
		headList.push_back(head);
		tailList.push_back(tail);
		relationList.push_back(num);
		trainLength.push_back(len);
		int *con=(int *)calloc(len,sizeof(int));
		int *conl=(int *)calloc(len,sizeof(int));
		int *conr=(int *)calloc(len,sizeof(int));
		for (int i = 0; i < len; i++) {
			con[i] = tmpp[i];
			conl[i] = lefnum - i;
			conr[i] = rignum - i;
			if (conl[i] >= limit) conl[i] = limit;
			if (conr[i] >= limit) conr[i] = limit;
			if (conl[i] <= -limit) conl[i] = -limit;
			if (conr[i] <= -limit) conr[i] = -limit;
			if (conl[i] > PositionMaxE1) PositionMaxE1 = conl[i];
			if (conr[i] > PositionMaxE2) PositionMaxE2 = conr[i];
			if (conl[i] < PositionMinE1) PositionMinE1 = conl[i];
			if (conr[i] < PositionMinE2) PositionMinE2 = conr[i];
		}
		trainLists.push_back(con);
		trainPositionE1.push_back(conl);
		trainPositionE2.push_back(conr);
	}
	fclose(f);

	printf("building training path\n");
	int num_2 = 0;
	int num_3 = 0;
	int num_0 = 0;
	int bags_id=0;
	vector<string> b_train;
	int index = 0;
	int error = 0;
	for (map<string,vector<int> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
	{
		b_train.push_back(it->first);
		bags_train_id[it->first] = bags_id;
		bags_id++;
	}

	for(int i = 0; i<b_train.size(); i++)
	{
		printf("%d, %s, %d\n",i,b_train[i].c_str(),bags_train_id[b_train[i]]);
	}

	f = fopen("../data/train.txt", "r");
	int test = 0;
	while (fscanf(f,"%s",buffer)==1)  {
		string e1 = buffer;
		fscanf(f,"%s",buffer);
		string e2 = buffer;
		fscanf(f,"%s",buffer);
		string head_s = (string)(buffer);
		int head = wordMapping[(string)(buffer)];
		fscanf(f,"%s",buffer);
		int tail = wordMapping[(string)(buffer)];
		string tail_s = (string)(buffer);
		fscanf(f,"%s",buffer);

		if(count(train_pair[head].begin() , train_pair[head].end() , tail)==0)
		{
			train_pair[head].push_back(tail);
		}
		if(count(train_pair_bags[make_pair(head,tail)].begin() , train_pair_bags[make_pair(head,tail)].end() , bags_train_id[e1+"\t"+e2+"\t"+(string)(buffer)])==0)
		{
			train_pair_bags[make_pair(head,tail)].push_back(bags_train_id[e1+"\t"+e2+"\t"+(string)(buffer)]);
		}

		if(count(test_train_pair[head].begin() , test_train_pair[head].end() , tail)==0)
		{
			test_train_pair[head].push_back(tail);
		}
		if(count(test_train_pair_bags[make_pair(head,tail)].begin() , test_train_pair_bags[make_pair(head,tail)].end() , bags_train_id[e1+"\t"+e2+"\t"+(string)(buffer)])==0)
		{
			test_train_pair_bags[make_pair(head,tail)].push_back(bags_train_id[e1+"\t"+e2+"\t"+(string)(buffer)]);
		}


		while (fscanf(f,"%s", buffer)==1) {
			std::string con = buffer;
			if (con=="###END###") break;
		}

	}
	fclose(f);

	for(int i = 0; i < b_train.size();i++)
	{
		//printf("%d\n",i);
		int flag = 1;
		string s = b_train[i];
		int id = bags_train[s][0];
		int head = headList[id];
		int tail = tailList[id];
		for(int j = 0;j<train_pair_bags[make_pair(head,tail)].size();j++)
		{
			int idd = train_pair_bags[make_pair(head,tail)][j];
			if (idd == i)
				flag = 0;
		}
		if(flag == 1)
		{
			printf("%d, %s, %d, %d, %d\n",i,b_train[i].c_str(),id,head,tail);
			for(int j = 0; j < train_pair_bags[make_pair(head,tail)].size();j++)
			{
				printf("%d ",train_pair_bags[make_pair(head,tail)][j]);
			}
			printf("\n");
		}
		error += flag;
	}

	printf("b_train size is %d, error is %d, bags_train size is %d\n",b_train.size(),error,bags_train.size());

	printf("start searching path\n");
	for (int i = 0; i < bags_train.size(); i++)
	{
		int id = bags_train[b_train[i].c_str()][0];
		int e1 = headList[id];
		int e2 = tailList[id];
		if(e1 == 0)
		{
			num_0++;
			continue;
		}
		if(e2 == 0)
		{
			num_0++;
			continue;
		}
		printf("%d, %d, %d, %d\n",i,train_pair[e1].size(),e1,e2);

		for(int j = 0; j < train_pair[e1].size();j++)
		{
			int e3 = train_pair[e1][j];
			if(e3 == 0)
			{
				continue;
			}
			//printf("%d, %d\n",j,train_pair[e3].size());

			for (int k = 0; k < train_pair[e3].size();k++)
			{
				int e4 = train_pair[e3][k];
				if (e4 != e2)
					continue;
				for (int jj = 0; jj < train_pair_bags[make_pair(e1,e3)].size();jj++)
				{
					int id_mid_1 = train_pair_bags[make_pair(e1,e3)][jj];
					string s_tmp = b_train[id_mid_1];
					int id_tmp = bags_train[s_tmp][0];
					if(relationList[id_tmp] == 99)
					{
						continue;
					}
					for (int kk = 0; kk < train_pair_bags[make_pair(e3,e2)].size();kk++)
					{
						int id_mid_2 = train_pair_bags[make_pair(e3,e2)][kk];
						s_tmp = b_train[id_mid_2];
						id_tmp = bags_train[s_tmp][0];
						if(relationList[id_tmp] == 99)
						{
							continue;
						}
						int *a = (int *)malloc(3* sizeof(int));
						a[0] = 2;
						a[1] = id_mid_1;
						a[2] = id_mid_2;
						train_path[b_train[i]].push_back(a);
						num_2++;
					}
				}
			}
		}

		/*for(int j = 0; j < bags_train.size();j++)
		{
			printf("%d\n",j);

			int id_mid_1 = bags_train[b_train[j]][0];
			if (headList[id_mid_1]!=e1)
				continue;
			int e3 = tailList[id_mid_1];
			for (int k = 0; k < bags_train.size();k++)
			{
				int id_mid_2 = bags_train[b_train[k]][0];
				if (headList[id_mid_2]!=e3)
					continue;
				if (tailList[id_mid_2]!=e2)
					continue;
				int *a = (int *)malloc(3* sizeof(int));
				a[0] = 2;
				a[1] = id_mid_1;
				a[2] = id_mid_2;
				train_path[b_train[i]].push_back(a);
				num_2++;
			}
		}*/

		/*for(int j = 0; j < bags_train.size();j++)
		{
			printf("3 %d\n",j);

			int id_mid_1 = bags_train[b_train[j]][0];
			if (headList[id_mid_1]!=e1)
				continue;
			int e3 = tailList[id_mid_1];
			for (int k = 0; k < bags_train.size();k++)
			{
				printf("32 %d\n",k);

				int id_mid_2 = bags_train[b_train[k]][0];
				if (headList[id_mid_2]!=e3)
					continue;
				int e4 = tailList[id_mid_2];
				for (int l = 0; l < bags_train.size();l++)
				{
					int id_mid_3 = bags_train[b_train[l]][0];
					if (headList[id_mid_3]!=e4)
						continue;
					if (tailList[id_mid_3]!=e2)
						continue;
					int *a = (int *)malloc(4* sizeof(int));
					a[0] = 3;
					a[1] = id_mid_1;
					a[2] = id_mid_2;
					a[3] = id_mid_3;
					train_path[b_train[i]].push_back(a);
					num_3++;
				}
			}
		}*/
	}
	printf("%d,%d\n",num_0,num_2);
	printf("print\n");
	FILE *fout = fopen("./train_path.txt", "w");
	for (map<string,vector<int*> >:: iterator it = train_path.begin(); it!=train_path.end(); it++)
	{
		vector<int*> b = it->second;
		string s = it->first;
		fprintf(fout, "%s\n", s.c_str());
		for (int i = 0; i < b.size(); i++)
		{
			int *a = b[i];
			for (int j = 0; j <= a[0]; j++)
			{
				fprintf(fout, "%d\t", a[j]);
			}
			fprintf(fout,"\n");
		}
		fprintf(fout,"%d\n",-1);
	}
	fclose(fout);


	f = fopen("../data/test.txt", "r");	
	while (fscanf(f,"%s",buffer)==1)  {
		string e1 = buffer;
		fscanf(f,"%s",buffer);
		string e2 = buffer;
		bags_test[e1+"\t"+e2].push_back(testheadList.size());
		fscanf(f,"%s",buffer);
		string head_s = (string)(buffer);
		int head = wordMapping[(string)(buffer)];
		fscanf(f,"%s",buffer);
		string tail_s = (string)(buffer);
		int tail = wordMapping[(string)(buffer)];
		fscanf(f,"%s",buffer);
		int num = relationMapping[(string)(buffer)];
		if (num == 100)
		{
			fscanf(f,"%s",buffer);
			int rel = relationMapping[(string)(buffer)];
			rel_pred[testheadList.size()] = rel;
		}
		int len = 0 , lefnum = 0, rignum = 0;
		std::vector<int> tmpp;
		while (fscanf(f,"%s", buffer)==1) {
			std::string con = buffer;
			if (con=="###END###") break;
			int gg = wordMapping[con];
			if (head_s == con) lefnum = len;
			if (tail_s == con) rignum = len;
			len++;
			tmpp.push_back(gg);
		}
		testheadList.push_back(head);
		testtailList.push_back(tail);
		testrelationList.push_back(num);
		testtrainLength.push_back(len);
		int *con=(int *)calloc(len,sizeof(int));
		int *conl=(int *)calloc(len,sizeof(int));
		int *conr=(int *)calloc(len,sizeof(int));
		for (int i = 0; i < len; i++) {
			con[i] = tmpp[i];
			conl[i] = lefnum - i;
			conr[i] = rignum - i;
			if (conl[i] >= limit) conl[i] = limit;
			if (conr[i] >= limit) conr[i] = limit;
			if (conl[i] <= -limit) conl[i] = -limit;
			if (conr[i] <= -limit) conr[i] = -limit;
			if (conl[i] > PositionMaxE1) PositionMaxE1 = conl[i];
			if (conr[i] > PositionMaxE2) PositionMaxE2 = conr[i];
			if (conl[i] < PositionMinE1) PositionMinE1 = conl[i];
			if (conr[i] < PositionMinE2) PositionMinE2 = conr[i];
		}
		testtrainLists.push_back(con);
		testPositionE1.push_back(conl);
		testPositionE2.push_back(conr);
	}
	fclose(f);
	printf("building testing path\n");
	num_2 = 0;
	num_3 = 0;
	num_0 = 0;
	bags_id=0;
	vector<string> b_test;
	index = 0;
	error = 0;
	for (map<string,vector<int> >:: iterator it = bags_test.begin(); it!=bags_test.end(); it++)
	{
		b_test.push_back(it->first);
		bags_test_id[it->first] = bags_id;
		bags_id++;
	}
	for(int i = 0; i<b_test.size(); i++)
	{
		printf("%d, %s, %d\n",i,b_test[i].c_str(),bags_test_id[b_test[i]]);
	}

	f = fopen("../data/test.txt", "r");
	test = 0;
	while (fscanf(f,"%s",buffer)==1)  {
		string e1 = buffer;
		fscanf(f,"%s",buffer);
		string e2 = buffer;
		fscanf(f,"%s",buffer);
		string head_s = (string)(buffer);
		int head = wordMapping[(string)(buffer)];
		fscanf(f,"%s",buffer);
		int tail = wordMapping[(string)(buffer)];
		string tail_s = (string)(buffer);
		fscanf(f,"%s",buffer);

		if(count(test_pair[head].begin() , test_pair[head].end() , tail)==0)
		{
			test_pair[head].push_back(tail);
		}
		if(count(test_pair_bags[make_pair(head,tail)].begin() , test_pair_bags[make_pair(head,tail)].end() , bags_test_id[e1+"\t"+e2])==0)
		{
			test_pair_bags[make_pair(head,tail)].push_back(bags_test_id[e1+"\t"+e2]);
		}


		while (fscanf(f,"%s", buffer)==1) {
			std::string con = buffer;
			if (con=="###END###") break;
		}

	}
	fclose(f);

	for(int i = 0; i < b_test.size();i++)
	{
		//printf("%d\n",i);
		int flag = 1;
		string s = b_test[i];
		int id = bags_test[s][0];
		int head = testheadList[id];
		int tail = testtailList[id];
		for(int j = 0;j<test_pair_bags[make_pair(head,tail)].size();j++)
		{
			int idd = test_pair_bags[make_pair(head,tail)][j];
			if (idd == i)
				flag = 0;
		}
		if(flag == 1)
		{
			printf("%d, %s, %d, %d, %d\n",i,b_test[i].c_str(),id,head,tail);
			for(int j = 0; j < test_pair_bags[make_pair(head,tail)].size();j++)
			{
				printf("%d ",test_pair_bags[make_pair(head,tail)][j]);
			}
			printf("\n");
		}
		error += flag;
	}

	printf("b_test size is %d, error is %d, bags_test size is %d\n",b_test.size(),error,bags_test.size());

	printf("start searching path\n");
	for (int i = 0; i < bags_test.size(); i++)
	{
		//printf("gg1\t");
		int id = bags_test[b_test[i].c_str()][0];
		int e1 = testheadList[id];
		int e2 = testtailList[id];
		//printf("gg2\t");	
		if(e1 == 0)
		{
			num_0++;
			continue;
		}
		if(e2 == 0)
		{
			num_0++;
			continue;
		}
		//printf("gg3\t");
		int id_mid_1 = test_pair_bags[make_pair(e1,e2)][0];
		//printf("gg4\t");
		string s_tmp = b_test[id_mid_1];
		//printf("gg5\t");
		int id_tmp = bags_test[s_tmp][0];
		//printf("gg6\n");
		//if(testrelationList[id_tmp] != 100)
		//{
		//	continue;
		//}
		printf("%d, %d, %d, %d\n",i,test_pair[e1].size(),e1,e2);

		for(int j = 0; j < test_pair[e1].size();j++)
		{
			int e3 = test_pair[e1][j];
			//printf("%d, %d\n",j,train_pair[e3].size());
			if(e3 == 0)
			{
				continue;
			}
			for (int k = 0; k < test_pair[e3].size();k++)
			{
				int e4 = test_pair[e3][k];
				if (e4 != e2)
					continue;
				for (int jj = 0; jj < test_pair_bags[make_pair(e1,e3)].size();jj++)
				{
					int id_mid_1 = test_pair_bags[make_pair(e1,e3)][jj];
					string s_tmp = b_test[id_mid_1];
					int id_tmp = bags_test[s_tmp][0];
					if(testrelationList[id_tmp] == 99 || testrelationList[id_tmp] == 100)
					{
						continue;
					}
					for (int kk = 0; kk < test_pair_bags[make_pair(e3,e2)].size();kk++)
					{
						int id_mid_2 = test_pair_bags[make_pair(e3,e2)][kk];
						s_tmp = b_test[id_mid_2];
						id_tmp = bags_test[s_tmp][0];
						if(testrelationList[id_tmp] == 99 || testrelationList[id_tmp] == 100)
						{
							continue;
						}
						int *a = (int *)malloc(5* sizeof(int));
						a[0] = 2;
						a[1] = id_mid_1;
						a[2] = 0;
						a[3] = id_mid_2;
						a[4] = 0;
						test_path[b_test[i]].push_back(a);
						num_2++;
					}
				}
			}

			for (int k = 0; k < test_train_pair[e3].size();k++)
			{
				int e4 = test_train_pair[e3][k];
				if (e4 != e2)
					continue;
				for (int jj = 0; jj < test_pair_bags[make_pair(e1,e3)].size();jj++)
				{
					int id_mid_1 = test_pair_bags[make_pair(e1,e3)][jj];
					string s_tmp = b_test[id_mid_1];
					int id_tmp = bags_test[s_tmp][0];
					if(testrelationList[id_tmp] == 99 || testrelationList[id_tmp] == 100)
					{
						continue;
					}
					for (int kk = 0; kk < test_train_pair_bags[make_pair(e3,e2)].size();kk++)
					{
						int id_mid_2 = test_train_pair_bags[make_pair(e3,e2)][kk];
						s_tmp = b_train[id_mid_2];
						id_tmp = bags_train[s_tmp][0];
						if(relationList[id_tmp] == 99)
						{
							continue;
						}
						int *a = (int *)malloc(5* sizeof(int));
						a[0] = 2;
						a[1] = id_mid_1;
						a[2] = 0;
						a[3] = id_mid_2;
						a[4] = 1;
						test_path[b_test[i]].push_back(a);
						num_2++;
					}
				}
			}
		}

		for(int j = 0; j < test_train_pair[e1].size();j++)
		{
			int e3 = test_train_pair[e1][j];
			//printf("%d, %d\n",j,train_pair[e3].size());
			if(e3 == 0)
			{
				continue;
			}
			for (int k = 0; k < test_train_pair[e3].size();k++)
			{
				int e4 = test_train_pair[e3][k];
				if (e4 != e2)
					continue;
				for (int jj = 0; jj < test_train_pair_bags[make_pair(e1,e3)].size();jj++)
				{
					int id_mid_1 = test_train_pair_bags[make_pair(e1,e3)][jj];
					string s_tmp = b_train[id_mid_1];
					int id_tmp = bags_train[s_tmp][0];
					if(relationList[id_tmp] == 99)
					{
						continue;
					}
					for (int kk = 0; kk < test_pair_bags[make_pair(e3,e2)].size();kk++)
					{
						int id_mid_2 = test_pair_bags[make_pair(e3,e2)][kk];
						s_tmp = b_test[id_mid_2];
						id_tmp = bags_test[s_tmp][0];
						if(testrelationList[id_tmp] == 99 || testrelationList[id_tmp] == 100)
						{
							continue;
						}
						int *a = (int *)malloc(5* sizeof(int));
						a[0] = 2;
						a[1] = id_mid_1;
						a[2] = 1;
						a[3] = id_mid_2;
						a[4] = 0;
						test_path[b_test[i]].push_back(a);
						num_2++;
					}
				}
			}

			for (int k = 0; k < test_train_pair[e3].size();k++)
			{
				int e4 = test_train_pair[e3][k];
				if (e4 != e2)
					continue;
				for (int jj = 0; jj < test_train_pair_bags[make_pair(e1,e3)].size();jj++)
				{
					int id_mid_1 = test_train_pair_bags[make_pair(e1,e3)][jj];
					string s_tmp = b_train[id_mid_1];
					int id_tmp = bags_train[s_tmp][0];
					if(relationList[id_tmp] == 99)
					{
						continue;
					}
					for (int kk = 0; kk < test_train_pair_bags[make_pair(e3,e2)].size();kk++)
					{
						int id_mid_2 = test_train_pair_bags[make_pair(e3,e2)][kk];
						s_tmp = b_train[id_mid_2];
						id_tmp = bags_train[s_tmp][0];
						if(relationList[id_tmp] == 99)
						{
							continue;
						}
						int *a = (int *)malloc(5* sizeof(int));
						a[0] = 2;
						a[1] = id_mid_1;
						a[2] = 1;
						a[3] = id_mid_2;
						a[4] = 1;
						test_path[b_test[i]].push_back(a);
						num_2++;
					}
				}
			}
		}

	}
	printf("%d,%d\n",num_0,num_2);
	printf("print");
	fout = fopen("./test_path.txt", "w");
	for (map<string,vector<int*> >:: iterator it = test_path.begin(); it!=test_path.end(); it++)
	{
		vector<int*> b = it->second;
		string s = it->first;
		fprintf(fout, "%s\n", s.c_str());
		for (int i = 0; i < b.size(); i++)
		{
			int *a = b[i];
			for (int j = 0; j <= 4; j++)
			{
				fprintf(fout, "%d\t", a[j]);
			}
			fprintf(fout,"\n");
		}
		fprintf(fout,"%d\n",-1);
	}
	fclose(fout);

	cout<<PositionMinE1<<' '<<PositionMaxE1<<' '<<PositionMinE2<<' '<<PositionMaxE2<<endl;

	for (int i = 0; i < trainPositionE1.size(); i++) {
		int len = trainLength[i];
		int *work1 = trainPositionE1[i];
		for (int j = 0; j < len; j++)
			work1[j] = work1[j] - PositionMinE1;
		int *work2 = trainPositionE2[i];
		for (int j = 0; j < len; j++)
			work2[j] = work2[j] - PositionMinE2;
	}

	for (int i = 0; i < testPositionE1.size(); i++) {
		int len = testtrainLength[i];
		int *work1 = testPositionE1[i];
		for (int j = 0; j < len; j++)
			work1[j] = work1[j] - PositionMinE1;
		int *work2 = testPositionE2[i];
		for (int j = 0; j < len; j++)
			work2[j] = work2[j] - PositionMinE2;
	}
	PositionTotalE1 = PositionMaxE1 - PositionMinE1 + 1;
	PositionTotalE2 = PositionMaxE2 - PositionMinE2 + 1;
}

float CalcTanh(float con) {
	if (con > 20) return 1.0;
	if (con < -20) return -1.0;
	float sinhx = exp(con) - exp(-con);
	float coshx = exp(con) + exp(-con);
	return sinhx / coshx;
}

float tanhDao(float con) {
	float res = CalcTanh(con);
	return 1 - res * res;
}
float sigmod(float con) {
	if (con > 20) return 1.0;
	if (con < -20) return 0.0;
	con = exp(con);
	return con / (1 + con);
}

int getRand(int l,int r) {
	int len = r - l;
	int res = rand()*rand() % len;
	if (res < 0)
		res+=len;
	return res + l;
}

float getRandU(float l, float r) {
	float len = r - l;
	float res = (float)(rand()) / RAND_MAX;
	return res * len + l;
}

void norm(float* a, int ll, int rr)
{
	float tmp = 0;
	for (int i=ll; i<rr; i++)
		tmp+=a[i]*a[i];
	if (tmp>1)
	{
		tmp = sqrt(tmp);
		for (int i=ll; i<rr; i++)
			a[i]/=tmp;
	}
}


int main()
{
	init();
	return 0;
}
