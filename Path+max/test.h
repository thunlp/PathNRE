#ifndef TEST_H
#define TEST_H
#include "init.h"
#include <algorithm>
#include <map>

int tipp = 0;
float ress = 0;
vector<string> b;
vector<string> a;
double tot;
vector<pair<int,double> > aa;

pthread_mutex_t mutexx;
vector<int> ll_test;


vector<double> cal_cnn(int *sentence, int *testPositionE1, int *testPositionE2, int len)
{
	int tip[dimensionC];
	double r[dimensionC];
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
				 	res += matrixW1[last + tot] * wordVec[last1+k];
				 	tot++;
				}
				int last2 = testPositionE1[j] * dimensionWPE;
				int last3 = testPositionE2[j] * dimensionWPE;
				for (int k = 0; k < dimensionWPE; k++) {
				 	res += matrixW1PositionE1[lastt + tot1] * positionVecE1[last2+k];
				 	res += matrixW1PositionE2[lastt + tot1] * positionVecE2[last3+k];
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
		r[i] = mx + matrixB1[i];
	}

	for (int i = 0; i < dimensionC; i++) {
		r[i] = CalcTanh(r[i]);
	}

	vector<double> res;
	double tmp = 0;
	for (int j = 0; j < relationTotal; j++) {
		float s = 0;
		for (int i = 0; i < dimensionC; i++)
			s +=  0.5 * matrixRelation[j * dimensionC + i] * r[i];
		s += matrixRelationPr[j];
		s = exp(s);
		tmp+=s;
		res.push_back(s);
	}
	for (int j = 0; j < relationTotal; j++) 
		res[j]/=tmp;
	return res;
}

vector<double> test(int bags_id) {
	vector<double> score_main;
	vector<double> score_sub;
	double max_score = 0.0;
	for (int i = 0; i < relationTotal; i++)
	{
		score_main.push_back(0.0);
		score_sub.push_back(0.0);
	}

	for(int i = 0; i < bags_test[b[bags_id]].size(); i++)
	{
		int sen_id = bags_test[b[bags_id]][i];
		vector<double> score_tmp = cal_cnn(testtrainLists[sen_id],  testPositionE1[sen_id], testPositionE2[sen_id], testtrainLength[sen_id]);
		for(int j = 0; j < relationTotal; j++)
		{
			score_main[j] = max(score_main[j],score_tmp[j]);
			max_score = max(max_score,score_main[j]);
		}

	}

	for(int i = 0; i < test_path[b[bags_id]].size(); i++)
	{
		int *path = test_path[b[bags_id]][i];
		float tmp0 = -1e8;
		int r1 = 0;
		int r2 = 0;
		if (path[2] == 0){
			for(int j = 0; j < bags_test[b[path[1]]].size(); j++)
			{
				int sen_id = bags_test[b[path[1]]][j];
				vector<double> score_tmp = cal_cnn(testtrainLists[sen_id],  testPositionE1[sen_id], testPositionE2[sen_id], testtrainLength[sen_id]);
				for (int k = 0; k < relationTotal; k++)
				{
					if(score_tmp[k]>tmp0)
					{
						tmp0 = score_tmp[k];
						r1 = k;
					}
				}
			}
		}
		if(path[2] == 1)
		{
			for(int j = 0; j < bags_train[a[path[1]]].size(); j++)
			{
				int sen_id = bags_train[a[path[1]]][j];
				tmp0 = 1;
				r1 = relationList[sen_id];
			}
		}
		float res1 = tmp0;
		tmp0 = -1e8;
		if(path[4]==0){
			for(int j = 0; j < bags_test[b[path[3]]].size(); j++)
			{
				int sen_id = bags_test[b[path[3]]][j];
				vector<double> score_tmp = cal_cnn(testtrainLists[sen_id],  testPositionE1[sen_id], testPositionE2[sen_id], testtrainLength[sen_id]);
				for (int k = 0; k < relationTotal; k++)
				{
					if(score_tmp[k]>tmp0)
					{
						tmp0 = score_tmp[k];
						r2 = k;
					}
				}
			}
		}
		if(path[4] == 1)
		{
			for(int j = 0; j < bags_train[a[path[3]]].size(); j++)
			{
				int sen_id = bags_train[a[path[3]]][j];
				tmp0 = 1;
				r2 = relationList[sen_id];
			}
		}
		float res2 = tmp0;
		for(int j = 0; j < relationTotal; j++)
		{
			double loss = relation_loss(j,r1,r2,1,0,0);
			score_sub[j] = max(score_sub[j],(1-max_score)*w*res1*res2*(loss));
		}

	}
	for(int j = 0; j < relationTotal; j++)
	{
		score_main[j] += score_sub[j];
	}
	return score_main;

}


bool cmp(pair<int,double> a,pair<int,double> b)
{
    return a.second>b.second;
}


void* testMode(void *id ) 
{
	int ll = ll_test[(long long)id];
	int rr;
	if ((long long)id==num_threads-1)
		rr = b.size();
	else
		rr = ll_test[(long long)id+1];
	for (int ii = ll; ii < rr; ii++)
	{

		vector<double> score;
		score = test(ii);
		map<int,int> ok;
		ok.clear();
		for(int j = 0; j < bags_test[b[ii]].size();j++)
		{
			int i = bags_test[b[ii]][j];
			ok[testrelationList[i]] = 1;
		}
		pthread_mutex_lock (&mutexx);
		for (int j = 0; j < relationTotal-1; j++) 
			aa.push_back(make_pair(ok.count(j),score[j]));
		pthread_mutex_unlock(&mutexx);
	}

}

double max_pre = 0;

void test() {
	for (int j = 0; j < relationTotal; j++) 
		cout<<matrixRelationPr[j]<<' ';
	cout<<endl;
	aa.clear();
	b.clear();
	tot = 0;
	ll_test.clear();
	vector<int> b_sum;
	b_sum.clear();
	for (map<string,vector<int> >:: iterator it = bags_test.begin(); it!=bags_test.end(); it++)
	{
		
		map<int,int> ok;
		ok.clear();
		for (int k=0; k<it->second.size(); k++)
		{
			int i = it->second[k];
			if (testrelationList[i]<99)
				ok[testrelationList[i]]=1;
		}
		tot+=ok.size();
		if (it->second.size()>0)
		{
			b.push_back(it->first);
			b_sum.push_back(it->second.size());
		}
	}
	for (map<string,vector<int> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
	{
		

		if (it->second.size()>0)
		{
			a.push_back(it->first);
		}
	}

	for (int i=1; i<b_sum.size(); i++)
		b_sum[i] += b_sum[i-1];
	cout<<b_sum[b_sum.size()-1]<<' '<<b_sum.size()-1<<endl;
	int now = 0;
	ll_test.resize(num_threads+1);
	for (int i=0; i<b_sum.size(); i++)
		if (b_sum[i]>=b_sum[b_sum.size()-1]/num_threads*now)
		{
			ll_test[now] = i;
			now+=1;
		}
	cout<<"tot:\t"<<tot<<endl;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	for (int a = 0; a < num_threads; a++)
		pthread_create(&pt[a], NULL, testMode,  (void *)a);
	for (int a = 0; a < num_threads; a++)
		pthread_join(pt[a], NULL);
	free(pt);
	cout<<"begin sort"<<' '<<aa.size()<<endl;
	sort(aa.begin(),aa.end(),cmp);
	double correct=0;
	int output_flag = 0;
	float correct1 = 0;
	for (int i=0; i<min(2000,int(aa.size())); i++)
	{
		if (aa[i].first!=0)
			correct1++;	
		float precision = correct1/(i+1);
		float recall = correct1/tot;
		if (i%100==0)
			cout<<"precision:\t"<<correct1/(i+1)<<'\t'<<"recall:\t"<<correct1/tot<<endl;
		if (recall>0.1&&precision>max_pre)
		{
			max_pre = precision;
			output_flag = 1;
		}
	}

	{
		FILE* f = fopen(("out/pr"+version+".txt").c_str(), "w");
		for (int i=0; i<min(20000,int(aa.size())); i++)
		{
			if (aa[i].first!=0)
				correct++;	
			fprintf(f,"%lf\t%lf\t%lf\n",correct/(i+1), correct/tot,aa[i].second);
		}
		fclose(f);
		if (!output_model)
			return;
		FILE *fout = fopen(("./out/matrixW1+B1.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\t%d\t%d\n", dimensionC, dimension, window, dimensionWPE);
		for (int i = 0; i < dimensionC; i++) {
			for (int j = 0; j < dimension * window; j++)
				fprintf(fout, "%f\t",matrixW1[i* dimension*window+j]);
			for (int j = 0; j < dimensionWPE * window; j++)
				fprintf(fout, "%f\t",matrixW1PositionE1[i* dimensionWPE*window+j]);
			for (int j = 0; j < dimensionWPE * window; j++)
				fprintf(fout, "%f\t",matrixW1PositionE2[i* dimensionWPE*window+j]);
			for (int j=0; j<3; j++)
				fprintf(fout, "%f\t", matrixB1[i*3+j]);
			fprintf(fout, "\n");
		}
		fclose(fout);

		fout = fopen(("./out/matrixRl.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\n", relationTotal, dimensionC);
		for (int i = 0; i < relationTotal; i++) {
			for (int j = 0; j < 3 * dimensionC; j++)
				fprintf(fout, "%f\t", matrixRelation[3 * i * dimensionC + j]);
			fprintf(fout, "\n");
		}
		for (int i = 0; i < relationTotal; i++) 
			fprintf(fout, "%f\t",matrixRelationPr[i]);
		fprintf(fout, "\n");
		fclose(fout);

		fout = fopen(("./out/matrixPosition.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\t%d\n", PositionTotalE1, PositionTotalE2, dimensionWPE);
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
	
		fout = fopen(("./out/word2vec.txt"+version).c_str(), "w");
		fprintf(fout,"%d\t%d\n",wordTotal,dimension);
		for (int i = 0; i < wordTotal; i++)
		{
			for (int j=0; j<dimension; j++)
				fprintf(fout,"%f\t",wordVec[i*dimension+j]);
			fprintf(fout,"\n");
		}
		fclose(fout);
	}
}

#endif