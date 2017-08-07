#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>
#include <map>
#include <cmath>
#include <pthread.h>
#include <iostream>


#include<assert.h>
#include<ctime>
#include<sys/time.h>

#include "init.h"
#include "test.h"

using namespace std;

double score = 0;
float alpha1 = 0;
vector<string> b_train;
vector<int> c_train;

struct timeval t_start,t_end; 
long start_time,end_time;

void time_begin()
{
  
  gettimeofday(&t_start, NULL); 
  start_time = ((long)t_start.tv_sec)*1000+(long)t_start.tv_usec/1000; 
}
void time_end()
{
  gettimeofday(&t_end, NULL); 
  end_time = ((long)t_end.tv_sec)*1000+(long)t_end.tv_usec/1000; 
  cout<<"time(s):\t"<<(double(end_time)-double(start_time))/1000<<endl;
}




double train(int flag, float g_bar, int *sentence, int *trainPositionE1, int *trainPositionE2, int len, int e1, int e2, int r1, float &res, float &res1, float *matrixW1Dao, float *matrixB1Dao, float *r, float *matrixRelationDao,
	float *positionVecDaoE1, float *positionVecDaoE2, float*matrixW1PositionE1Dao, float*matrixW1PositionE2Dao,  float alpha) {
		int tip[dimensionC];
			
		for (int i = 0; i < dimensionC; i++) {
			int last = i * dimension * window;
			int lastt = i * dimensionWPE * window;
			float mx = -FLT_MAX;
			for (int i1 = 0; i1 <= len - window; i1++) {
				float res = 0;
				int tot = 0;
				int tot1 = 0;
				for (int j = i1; j < i1 + window; j++)  
				if (j>=0&&j<len){
					int last1 = sentence[j] * dimension;
				 	for (int k = 0; k < dimension; k++) {
				 		res += matrixW1Dao[last + tot] * wordVecDao[last1+k];
				 		tot++;
				 	}
				 	int last2 = trainPositionE1[j] * dimensionWPE;
				 	int last3 = trainPositionE2[j] * dimensionWPE;
				 	for (int k = 0; k < dimensionWPE; k++) {
				 		res += matrixW1PositionE1Dao[lastt + tot1] * positionVecDaoE1[last2+k];
				 		res += matrixW1PositionE2Dao[lastt + tot1] * positionVecDaoE2[last3+k];
				 		tot1++;
				 	}
				}
				else
				{
					tot+=dimension;
					tot1+=dimensionWPE;
				}
				if (res > mx) {
					mx = res;
					tip[i] = i1;
				}
			}
			r[i] = mx + matrixB1Dao[i];
		}

		for (int i = 0; i < dimensionC; i++) {
			r[i] = CalcTanh(r[i]);
		}
		
		vector<int> dropout;
		for (int i = 0; i < dimensionC; i++) 
			dropout.push_back(rand()%2);
		
		vector<double> f_r;	
		double sum = 0;
		for (int j = 0; j < relationTotal; j++) {
			float s = 0;
			for (int i = 0; i < dimensionC; i++) {
				s += dropout[i] * r[i] * matrixRelationDao[j * dimensionC + i];
			}
			s += matrixRelationPrDao[j];
			f_r.push_back(exp(s));
			sum+=f_r[j];
		}
		double rt = f_r[r1]/sum;
		if (flag)
		{
			float s1, g, s2;
			for (int i = 0; i < dimensionC; i++) {
				if (dropout[i]==0)
					continue;
				int last = i * dimension * window;
				int tot = 0;
				int lastt = i * dimensionWPE * window;
				int tot1 = 0;
				float g1 = 0;
				for (int r2 = 0; r2<relationTotal; r2++)
				{
					g = f_r[r2]/sum*alpha;
					if (r2 == r1)
						g -= alpha;
					g = g*g_bar;
					g1 += g * matrixRelationDao[r2 * dimensionC + i] * (1 -  r[i] * r[i]);
					matrixRelation[r2 * dimensionC + i] -= g * r[i];
					if (i==0)
						matrixRelationPr[r2] -= g;
				}
				for (int j = 0; j < window; j++)  
				if (tip[i]+j>=0&&tip[i]+j<len){
					int last1 = sentence[tip[i] + j] * dimension;
					for (int k = 0; k < dimension; k++) {
						matrixW1[last + tot] -= g1 * wordVecDao[last1+k];
						wordVec[last1 + k] -= g1 * matrixW1Dao[last + tot];
						tot++;
					}
					int last2 = trainPositionE1[tip[i] + j] * dimensionWPE;
					int last3 = trainPositionE2[tip[i] + j] * dimensionWPE;
					for (int k = 0; k < dimensionWPE; k++) {
						matrixW1PositionE1[lastt + tot1] -= g1 * positionVecDaoE1[last2 + k];
						matrixW1PositionE2[lastt + tot1] -= g1 * positionVecDaoE2[last3 + k];
						positionVecE1[last2 + k] -= g1 * matrixW1PositionE1Dao[lastt + tot1];
						positionVecE2[last3 + k] -= g1 * matrixW1PositionE2Dao[lastt + tot1];
						tot1++;
					}
				}
				matrixB1[i] -= g1;
			}

			for (int i = 0; i < dimensionC; i++) {
				int last = dimension * window * i;
				res1+=Belt * matrixB1Dao[i] * matrixB1Dao[i];

				for (int j = dimension * window -1; j >= 0; j--) {
					res1+= Belt * matrixW1Dao[last + j] * matrixW1Dao[last + j];
					matrixW1[last + j] += - Belt * matrixW1Dao[last + j] * alpha * 2; 
				}

				last = dimensionWPE * window * i;
				for (int j = dimensionWPE * window -1; j>=0; j--) {
					matrixW1PositionE1[last + j] += -Belt * matrixW1PositionE1Dao[last + j] * alpha * 2;
					matrixW1PositionE2[last + j] += -Belt * matrixW1PositionE2Dao[last + j] * alpha * 2;
				}

				matrixB1[i] += -Belt * matrixB1Dao[i] *alpha * 2;
			}
		}
		return rt;
}

int turn;

double path_cnn (int bags_id, float *matrixW1Dao, float *matrixB1Dao, float *r, float *matrixRelationDao,
	float *positionVecDaoE1, float *positionVecDaoE2, float*matrixW1PositionE1Dao, float*matrixW1PositionE2Dao,  float alpha)
{

	double tmp1 = -1e8;
	int tmp2 = -1;
	float res = 0;
	float res1 = 0;
	vector<double> chain_loss;
	vector<int> chain_id; 
	vector<double> chain_relation_loss;

	for (int k=0; k<bags_train[b_train[bags_id]].size(); k++)//softmax on main sentence
	{
		int i = bags_train[b_train[bags_id]][k];
		double tmp = train(0,1.0,trainLists[i], trainPositionE1[i], trainPositionE2[i], trainLength[i], headList[i], tailList[i], relationList[i], res, res1, matrixW1Dao, matrixB1Dao, r, matrixRelationDao, 
		positionVecDaoE1, positionVecDaoE2, matrixW1PositionE1Dao, matrixW1PositionE2Dao, alpha1);
		if (tmp1<tmp)
		{
			tmp1 = tmp;
			tmp2 = i;
		}
	}
	chain_loss.push_back(tmp1);
	chain_id.push_back(tmp2);
	chain_relation_loss.push_back(1.0);

	int num = train_path[b_train[bags_id]].size();

	for(int j = 0; j < num; j++)//softmax on peripheral path
	{
		
		int *a = train_path[b_train[bags_id]][j];
		for(int l = 1; l<=2 ; l++)
		{
			int pair_id = a[l];
			tmp1 = -1e8;
			tmp2 = -1;
			for (int k=0; k<bags_train[b_train[pair_id]].size(); k++)//softmax on main sentence
			{
				int i = bags_train[b_train[pair_id]][k];
				double tmp = train(0,1.0,trainLists[i], trainPositionE1[i], trainPositionE2[i], trainLength[i], headList[i], tailList[i], relationList[i], res, res1, matrixW1Dao, matrixB1Dao, r, matrixRelationDao, 
				positionVecDaoE1, positionVecDaoE2, matrixW1PositionE1Dao, matrixW1PositionE2Dao, alpha1);
				if (tmp1<tmp)
				{
					tmp1 = tmp;
					tmp2 = i;
				}
			}
			chain_loss.push_back(tmp1);
			chain_id.push_back(tmp2);
		}
		double loss = relation_loss(relationList[chain_id[0]],relationList[chain_id[2*j+1]],relationList[chain_id[2*j+2]],0,0,0);
		chain_relation_loss.push_back(loss);
	}
	int max_path_id = 0;
	double max_add = -1e8;
	for (int j = 0; j < num; j++)
	{
		double tmp_add = chain_loss[2*j+1]*chain_loss[2*j+2]*chain_relation_loss[j+1];
		if (tmp_add > max_add)
		{
			max_add = tmp_add;
			max_path_id = j;
		}
	}
	
	double rt = 0;
	rt = rt + chain_loss[0];
	if (num>0)
		rt += (1-chain_loss[0])*w*chain_loss[2*max_path_id+1]*chain_loss[2*max_path_id+2]*(chain_relation_loss[max_path_id+1]);

	double nuissance = train(1,1*chain_loss[0]/rt,trainLists[chain_id[0]], trainPositionE1[chain_id[0]], trainPositionE2[chain_id[0]], trainLength[chain_id[0]], headList[chain_id[0]], tailList[chain_id[0]], relationList[chain_id[0]], res, res1, matrixW1Dao, matrixB1Dao, r, matrixRelationDao, 
	positionVecDaoE1, positionVecDaoE2, matrixW1PositionE1Dao, matrixW1PositionE2Dao, alpha1);
	
	if (num>0)
	{
		int j = max_path_id;
		nuissance = train(1,(1-chain_loss[0])*w*chain_loss[2*max_path_id+1]*chain_loss[2*max_path_id+2]*(chain_relation_loss[max_path_id+1])/rt,trainLists[chain_id[2*max_path_id+1]], trainPositionE1[chain_id[2*max_path_id+1]], trainPositionE2[chain_id[2*max_path_id+1]], trainLength[chain_id[2*max_path_id+1]], headList[chain_id[2*max_path_id+1]], tailList[chain_id[2*max_path_id+1]], relationList[chain_id[2*max_path_id+1]], res, res1, matrixW1Dao, matrixB1Dao, r, matrixRelationDao, 
						positionVecDaoE1, positionVecDaoE2, matrixW1PositionE1Dao, matrixW1PositionE2Dao, alpha1);
		nuissance = train(1,(1-chain_loss[0])*w*chain_loss[2*max_path_id+1]*chain_loss[2*max_path_id+2]*(chain_relation_loss[max_path_id+1])/rt,trainLists[chain_id[2*max_path_id+2]], trainPositionE1[chain_id[2*max_path_id+2]], trainPositionE2[chain_id[2*max_path_id+2]], trainLength[chain_id[2*max_path_id+2]], headList[chain_id[2*max_path_id+2]], tailList[chain_id[2*max_path_id+2]], relationList[chain_id[2*max_path_id+2]], res, res1, matrixW1Dao, matrixB1Dao, r, matrixRelationDao, 
						positionVecDaoE1, positionVecDaoE2, matrixW1PositionE1Dao, matrixW1PositionE2Dao, alpha1);
		int r0 = relationList[chain_id[0]];
		int r1 = relationList[chain_id[2*j+1]];
		int r2 = relationList[chain_id[2*j+2]];
		nuissance = relation_loss(r0,r1,r2,0,1,(1-chain_loss[0])*w*chain_loss[2*j+1]*chain_loss[2*j+2]*chain_relation_loss[j+1]/rt);

	}

	return rt;
}



double score_tmp = 0, score_max = 0;
pthread_mutex_t mutex1;
void* trainMode(void *id ) {
		unsigned long long next_random = (long long)id;
		float *r = (float *)calloc(dimensionC, sizeof(float));
		for(int i = 0; i < dimensionC; i++)
		{
			r[i] = 0.0;
		}
		{
				for (int k1 = batch; k1 > 0; k1--)
				{
					int j = getRand(0, c_train.size());
					j = c_train[j];
					score += path_cnn(j, matrixW1Dao, matrixB1Dao, r, matrixRelationDao, 
						positionVecDaoE1, positionVecDaoE2, matrixW1PositionE1Dao, matrixW1PositionE2Dao, alpha1);
					
				}
		}
		free(r);
}

void train() {
	int tmp = 0;
	b_train.clear();
	c_train.clear();
	for (map<string,vector<int> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
	{
		for (int i=0; i<max(1,1); i++)
			c_train.push_back(b_train.size());
		b_train.push_back(it->first);
		tmp+=it->second.size();
	}
	cout<<c_train.size()<<endl;

	float con = sqrt(6.0/(dimensionC+relationTotal));
	float con1 = sqrt(6.0/((dimensionWPE+dimension)*window));
	float con2 = sqrt(6.0/relationTotal);
	matrixRelation = (float *)calloc(dimensionC * relationTotal, sizeof(float));
	vectorRelation = (float *)calloc(relationTotal * dimensionR, sizeof(float));
	matrixRelationPr = (float *)calloc(relationTotal, sizeof(float));
	matrixRelationPrDao = (float *)calloc(relationTotal, sizeof(float));
	wordVecDao = (float *)calloc(dimension * wordTotal, sizeof(float));
	positionVecE1 = (float *)calloc(PositionTotalE1 * dimensionWPE, sizeof(float));
	positionVecE2 = (float *)calloc(PositionTotalE2 * dimensionWPE, sizeof(float));
	
	matrixW1 = (float*)calloc(dimensionC * dimension * window, sizeof(float));
	matrixW1PositionE1 = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
	matrixW1PositionE2 = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
	matrixB1 = (float*)calloc(dimensionC, sizeof(float));

	for (int i = 0; i < dimensionC; i++) {
		int last = i * window * dimension;
		for (int j = dimension * window - 1; j >=0; j--)
			matrixW1[last + j] = getRandU(-con1, con1);
		last = i * window * dimensionWPE;
		float tmp1 = 0;
		float tmp2 = 0;
		for (int j = dimensionWPE * window - 1; j >=0; j--) {
			matrixW1PositionE1[last + j] = getRandU(-con1, con1);
			tmp1 += matrixW1PositionE1[last + j]  * matrixW1PositionE1[last + j] ;
			matrixW1PositionE2[last + j] = getRandU(-con1, con1);
			tmp2 += matrixW1PositionE2[last + j]  * matrixW1PositionE2[last + j] ;
		}
		matrixB1[i] = getRandU(-con1, con1);
	}

	for (int i = 0; i < relationTotal; i++) 
	{
		matrixRelationPr[i] = getRandU(-con, con);				//add
		for (int j = 0; j < dimensionC; j++)
			matrixRelation[i * dimensionC + j] = getRandU(-con, con);
	}

	for(int i = 0; i < relationTotal; i++)	//initializing relation embedding
	{
		for (int j = 0; j < dimensionR; j++)
		{
			vectorRelation[i*dimensionR+j] = getRandU(-con2,con2);
		}
	}

	for (int i = 0; i < PositionTotalE1; i++) {
		float tmp = 0;
		for (int j = 0; j < dimensionWPE; j++) {
			positionVecE1[i * dimensionWPE + j] = getRandU(-con1, con1);
			tmp += positionVecE1[i * dimensionWPE + j] * positionVecE1[i * dimensionWPE + j];
		}
	}

	for (int i = 0; i < PositionTotalE2; i++) {
		float tmp = 0;
		for (int j = 0; j < dimensionWPE; j++) {
			positionVecE2[i * dimensionWPE + j] = getRandU(-con1, con1);
			tmp += positionVecE2[i * dimensionWPE + j] * positionVecE2[i * dimensionWPE + j];
		}
	}

	matrixRelationDao = (float *)calloc(dimensionC*relationTotal, sizeof(float));
	matrixW1Dao =  (float*)calloc(dimensionC * dimension * window, sizeof(float));
	matrixB1Dao =  (float*)calloc(dimensionC, sizeof(float));
	vectorRelationDao = (float *)calloc(relationTotal * dimensionR, sizeof(float));	

	positionVecDaoE1 = (float *)calloc(PositionTotalE1 * dimensionWPE, sizeof(float));
	positionVecDaoE2 = (float *)calloc(PositionTotalE2 * dimensionWPE, sizeof(float));
	matrixW1PositionE1Dao = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
	matrixW1PositionE2Dao = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
	printf("start sgd\n");
	for (turn = 0; turn < trainTimes; turn ++) {

		len = c_train.size();
		npoch  =  len / (batch * num_threads);
		alpha1 = alpha*rate/batch;

		score = 0;
		score_max = 0;
		score_tmp = 0;
		double score1 = score;
		int k_tmp = 0;
		printf("turn is %d\n",turn);
		time_begin();
		for (int k = 1; k <= npoch; k++) {
			score_max += batch * num_threads;
			memcpy(positionVecDaoE1, positionVecE1, PositionTotalE1 * dimensionWPE* sizeof(float));
			memcpy(positionVecDaoE2, positionVecE2, PositionTotalE2 * dimensionWPE* sizeof(float));
			memcpy(matrixW1PositionE1Dao, matrixW1PositionE1, dimensionC * dimensionWPE * window* sizeof(float));
			memcpy(matrixW1PositionE2Dao, matrixW1PositionE2, dimensionC * dimensionWPE * window* sizeof(float));
			memcpy(wordVecDao, wordVec, dimension * wordTotal * sizeof(float));
			memcpy(vectorRelationDao,vectorRelation,relationTotal*dimensionR* sizeof(float));

			memcpy(matrixW1Dao, matrixW1, sizeof(float) * dimensionC * dimension * window);
			memcpy(matrixB1Dao, matrixB1, sizeof(float) * dimensionC);
			memcpy(matrixRelationPrDao, matrixRelationPr, relationTotal * sizeof(float));				//add
			memcpy(matrixRelationDao, matrixRelation, dimensionC*relationTotal * sizeof(float));
			pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
			for (int a = 0; a < num_threads; a++)
				pthread_create(&pt[a], NULL, trainMode,  (void *)a);
			for (int a = 0; a < num_threads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
			if (k%(npoch/5)==0)
			{
				cout<<"npoch:\t"<<k<<'/'<<npoch<<endl;
				time_end();
				time_begin();
				cout<<"score:\t"<<(score-score1)/double(k-k_tmp)/double(batch)<<' '<<score_tmp<<endl;
				score1 = score;
				k_tmp = k;
			}
		}
		printf("Total Score:\t%f\n",score);
		printf("test\n");
	}
	//test();
	cout<<"Train End"<<endl;

	FILE *fout = fopen("./out/matrixW1+B1.txt", "w");
	for (int i = 0; i < dimensionC; i++) {
		for (int j = 0; j < dimension * window; j++)
			fprintf(fout, "%f\t",matrixW1[i* dimension*window+j]);
		for (int j = 0; j < dimensionWPE * window; j++)
			fprintf(fout, "%f\t",matrixW1PositionE1[i* dimensionWPE*window+j]);
		for (int j = 0; j < dimensionWPE * window; j++)
			fprintf(fout, "%f\t",matrixW1PositionE2[i* dimensionWPE*window+j]);
		fprintf(fout, "%f\n", matrixB1[i]);
	}
	fclose(fout);

	fout = fopen("./out/matrixRl.txt", "w");
	for (int i = 0; i < relationTotal; i++) {
		for (int j = 0; j < dimensionC; j++)
			fprintf(fout, "%f\t", matrixRelation[i * dimensionC + j]);
		fprintf(fout, "\n");
	}
	for (int i = 0; i < relationTotal; i++)
		fprintf(fout,"%f\t",matrixRelationPr[i]);
	fclose(fout);

	fout = fopen("./out/vectorRl.txt","w");
	for(int i = 0; i < relationTotal; i++)
	{
		for(int j = 0; j < dimensionR; j++)
		{
			fprintf(fout, "%f\t", vectorRelation[i * dimensionR + j]);
		}
		fprintf(fout,"\n");
	}
	fclose(fout);

	fout = fopen("./out/matrixPosition.txt", "w");
	for (int i = 0; i < PositionTotalE1; i++) {
		for (int j = 0; j < dimensionWPE; j++)
			fprintf(fout, "%f\t", positionVecE1[i * dimensionWPE + j]);
		fprintf(fout, "\n");
	}
	for (int i = 0; i < PositionTotalE2; i++) {
		for (int j = 0; j < dimensionWPE; j++)
			fprintf(fout, "%f\t", positionVecE2[i * dimensionWPE + j]);
		fprintf(fout, "\n");
	}
	fclose(fout);
	fout = fopen("./out/word2vec.txt", "w");
	for (int i = 0; i < wordTotal; i++)
	{
		for (int j=0; j<dimension; j++)
			fprintf(fout,"%f\t",wordVec[i*dimension+j]);
		fprintf(fout,"\n");
	}
	fclose(fout);
}

int main(int argc, char ** argv) {
	logg = fopen("log.txt","w");
	cout<<"Init Begin."<<endl;
	init();
	cout<<bags_train.size()<<' '<<bags_test.size()<<endl;
	cout<<"Init End."<<endl;
	train();
	fclose(logg);
	return 0;
}
