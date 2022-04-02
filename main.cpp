#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "RedeNeural.hpp"
#include "utils.hpp"

using namespace std;

//#define epochs 50
#define epochs 500

int main(){
	vector<pair<vector<double>, vector<double>>> elements;

	ifstream infile;
	infile.open("dataset.txt", ios::in);

	double a,b,c,d;
	string nameClass;

	int count = 0;

	int len_train = 60;



	//RedeNeural *rede = new RedeNeural(4,4,50,3);
	RedeNeural *rede = new RedeNeural(4,3,50,3);
	while (infile >> a >> b >> c >> d >> nameClass){
		vector<double> values;
		vector<double> output;

		values.push_back(a);
		values.push_back(b);
		values.push_back(c);
		values.push_back(d);

		if(nameClass == "Iris-virginica"){
			output.push_back(1.0);
			output.push_back(0.0);
			output.push_back(0.0);
		}
		else if(nameClass == "Iris-setosa"){
			output.push_back(0.0);
			output.push_back(1.0);
			output.push_back(0.0);
		}
		else{
			output.push_back(0.0);
			output.push_back(0.0);
			output.push_back(1.0);
		}

		elements.push_back(make_pair(values,output));

		//cout << values[0] << " " << values[1] << " " << values[2] << " " << values[3] << " " << nameClass << " " << endl;
		//cout << output[0] << " " << output[1] << " " << output[2] << endl << endl;
	}
	cout << "Tamanho do vetor" << elements.size() << endl;

	double input[4];
	double output[3];


	for(int i = 0; i < epochs; i++){
		for(int j=0; j< len_train; j++)
		{	
			double input[4];
			double output[3];
			copy(elements[j].first.begin(), elements[j].first.end(), input);
			copy(elements[j].second.begin(), elements[j].second.end(), output);
			//cout << input[0] << " " << input[1] << " " << input[2] << " " << input[3] << " " << endl;
			rede->RNA_backPropagation(input, output);
		}
	}

	int acc = 0;
	int len_test = 150 - len_train;

	for(int i = len_train; i < 150; i++){
		double input[4];
		copy(elements[i].first.begin(), elements[i].first.end(), input);
		double output[3];
		rede->RNA_copiarParaEntrada(input);
		rede->RNA_calcularSaida();
		rede->RNA_copiarParaSaida(output);

		cout << "Valor Predito: "<< output[0] << " " << output[1] << " " << output[2] << endl;
		cout << "Valor Real: "<< elements[i].second[0] << " " << elements[i].second[1] << " " << elements[i].second[2] << endl << endl;

		double realValue[3];
		copy(elements[i].second.begin(), elements[i].second.end(), realValue);

		if(returnIndexGreatElement(output,3) == returnIndexGreatElement(realValue,3))
			acc++;
	}
	cout << "Total: " << len_test << endl;
	cout << "Acc: " << acc << endl;

	infile.close();

	return 0;
}