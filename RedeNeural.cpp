#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>
#include <random>
#include "RedeNeural.hpp"

#define Ativacao(X)         LeakyRELU(X)
#define Derivada(X)         LeakyRELUDx(X)


#define AtivacaoOcultas(X)        LeakyRELU(X)
#define AtivacaoSaida(X)          RELU(X)

#define alfa 0.01

int global_count = 0;


//Global random variables
random_device rd; 
mt19937 gen(rd()); 



double LeakyRELU(double X)
{
    if(X <= 0.0)
    {
        return (alfa * X);
    }
    else
    {
        return X;
    }

}

double LeakyRELUDx(double X)
{
    if(X < 0.0)
    {
        return alfa;
    }
    else
    {
        return 1.0;
    }
}



double RELU(double X)
{
    if(X < 0)
    {
        return 0;
    }
    else
    {
        if(X < 1)
        {
            return X;
        }
        else
        {
            return 1;
        }
    }

}

double RELUDx(double X)
{
    if(X < 0)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}


Neuronio::Neuronio(int quantidadeLigacoes){
    this->quantidadeLigacoes = quantidadeLigacoes;
    this->peso = new double[quantidadeLigacoes];

    for(int i = 0; i < quantidadeLigacoes; i++){
        normal_distribution<double> d(0, 1);
        this->peso[i] = d(gen);
        //cout << "Peso original:" << peso[i] << endl;
        global_count++;
    }

    this->erro = 0.0;
    this->saida = 1.0;
}

Neuronio::Neuronio(){

}
/*
Neuronio::~Neuronio(void){
    delete []peso;
}
*/


Camada::Camada(){

}

Camada::Camada(int quantidadeNeuronios, int quantidadeLigacoes){
    this->quantidadeNeuronios = quantidadeNeuronios;
    this->neuronios = new Neuronio[quantidadeNeuronios];

    for(int i = 0; i < quantidadeNeuronios; i++)
        this->neuronios[i] = Neuronio(quantidadeLigacoes);
}

/*
Camada::~Camada(void){
    delete []neuronios;
}
*/




RedeNeural::RedeNeural(int qtdNeuroniosEntrada, int quantidadeEscondidas, int qtdNeuroniosEscondida, int qtdNeuroniosSaida){
    qtdNeuroniosEntrada = qtdNeuroniosEntrada + BIAS;
    qtdNeuroniosEscondida = qtdNeuroniosEscondida + BIAS;

    this->quantidadeEscondidas = quantidadeEscondidas;

    this->camadaEntrada = Camada(qtdNeuroniosEntrada,0);

    this->camadaEscondida = new Camada[quantidadeEscondidas];

    for(int i = 0; i < quantidadeEscondidas; i++){
        if(i == 0)
            this->camadaEscondida[i] = Camada(qtdNeuroniosEscondida,qtdNeuroniosEntrada);
        else
            this->camadaEscondida[i] = Camada(qtdNeuroniosEscondida,qtdNeuroniosEscondida);
    }

    this->camadaSaida = Camada(qtdNeuroniosSaida,qtdNeuroniosEscondida);
}

/*
RedeNeural::~RedeNeural(void){
    delete []camadaEscondida;
}
*/


void RedeNeural::RNA_copiarParaEntrada(double *vetor){
    for(int i = 0; i < this->camadaEntrada.quantidadeNeuronios - BIAS; i++)
        this->camadaEntrada.neuronios[i].saida = vetor[i];
}

void RedeNeural::RNA_copiarParaSaida(double *vetor){
    for(int i = 0; i < this->camadaSaida.quantidadeNeuronios; i++)
        vetor[i] = this->camadaSaida.neuronios[i].saida;
}

void RedeNeural::RNA_calcularSaida(void){
    int i,j,k;
    double somatorio;

    //Calculando a sa�da entre a camada de entrada e a primeira camada escondida
    for(i = 0; i < this->camadaEscondida[0].quantidadeNeuronios - BIAS; i++){
        somatorio = 0.0;
        for(j = 0; j < this->camadaEntrada.quantidadeNeuronios; j++)
            somatorio += (this->camadaEscondida[0].neuronios[i].peso[j]) * (this->camadaEntrada.neuronios[j].saida);
        this->camadaEscondida[0].neuronios[i].saida = AtivacaoOcultas(somatorio);
    }


    //Calculando entre a camada escondida i at� a i - 1
    for(k = 1; k < this->quantidadeEscondidas; k++){
        for(i = 0; i < this->camadaEscondida[k].quantidadeNeuronios - BIAS; i++){
            somatorio = 0.0;
            for(j = 0; j < this->camadaEscondida[k - 1].quantidadeNeuronios; j++)
                somatorio += (this->camadaEscondida[k - 1].neuronios[j].saida) * (this->camadaEscondida[k].neuronios[i].peso[j]);
            this->camadaEscondida[k].neuronios[i].saida = AtivacaoOcultas(somatorio);
        }
    }
    //Calculando entre a �ltima camada oculta e a camada de sa�da
    for(i = 0; i < this->camadaSaida.quantidadeNeuronios; i++){
        somatorio = 0.0;
        for(j = 0; j < this->camadaEscondida[k - 1].quantidadeNeuronios; j++)
            somatorio += (this->camadaEscondida[k - 1].neuronios[j].saida) * (this->camadaSaida.neuronios[i].peso[j]);
        this->camadaSaida.neuronios[i].saida = AtivacaoSaida(somatorio);
    }
}

int RedeNeural::RNA_backPropagation(double *entrada, double *saidaEsperada){
    int i,j,k;
    double somatorio;
    double erro = 0.0;
    double saida[3];
    //double saida[1];

    RNA_copiarParaEntrada(entrada);
    RNA_calcularSaida();

   for(i = 0; i < this->camadaSaida.quantidadeNeuronios; i++)
   {
       saida[i] = this->camadaSaida.neuronios[i].saida;
       //cout << saida[i] << endl;
       erro += 0.5 *(saidaEsperada[i] - saida[i]) * (saidaEsperada[i] - saida[i]);
   }

   if(erro > 0.0)
   {
       /// Setando erro da camada de saida ///////////////////////

       for(i=0; i<this->camadaSaida.quantidadeNeuronios; i++)
       {
           this->camadaSaida.neuronios[i].erro = saidaEsperada[i] - this->camadaSaida.neuronios[i].saida;
       }

       //////////////////////////////////////////////////////////////////
       /// Atualizando erro entre ultima camada escondida e camada saida ///////////////////////

       for(j = 0; j < this->camadaEscondida[this->quantidadeEscondidas - 1].quantidadeNeuronios; j++)
       {
           somatorio = 0.0;
           for(k=0; k<this->camadaSaida.quantidadeNeuronios; k++)
           {
               somatorio = somatorio + this->camadaSaida.neuronios[k].erro * this->camadaSaida.neuronios[k].peso[j];
           }
           this->camadaEscondida[this->quantidadeEscondidas-1].neuronios[j].erro = somatorio;
       }

       //////////////////////////////////////////////////////////////////
       /// Atualizando erro entre a camada escondida 'n' e 'n+1' ///////////////////////

       for(i=this->quantidadeEscondidas-2; i>=0; i--)
       {
           for(j=0; j<this->camadaEscondida[i].quantidadeNeuronios; j++)
           {
               somatorio = 0.0;
               for(k=0; k<this->camadaEscondida[i+1].quantidadeNeuronios; k++)
               {
                   somatorio = somatorio + this->camadaEscondida[i+1].neuronios[k].erro * this->camadaEscondida[i+1].neuronios[k].peso[j];
               }
               this->camadaEscondida[i].neuronios[j].erro = somatorio;
           }
       }
       /// Atualizando pesos camada entrada com primeira escondida////////////////////////

       for(j=0; j<this->camadaEscondida[0].quantidadeNeuronios; j++)
       {
           for(k=0; k<this->camadaEntrada.quantidadeNeuronios; k++)
           {
               this->camadaEscondida[0].neuronios[j].peso[k] = this->camadaEscondida[0].neuronios[j].peso[k] + (TAXA_APRENDIZADO*
                                                                                                                this->camadaEscondida[0].neuronios[j].erro*
                                                                                                                this->camadaEntrada.neuronios[k].saida*
                                                                                                                Derivada(this->camadaEscondida[0].neuronios[j].saida));
           }
       }

       /// Atualizando pesos camada 'n' com 'n-1'////////////////////////
       for(i=1; i<this->quantidadeEscondidas; i++)
       {
           for(j=0; j<this->camadaEscondida[i].quantidadeNeuronios; j++)
           {
               for(k=0; k<this->camadaEscondida[i-1].quantidadeNeuronios; k++)
               {
                   this->camadaEscondida[i].neuronios[j].peso[k] = this->camadaEscondida[i].neuronios[j].peso[k] + (TAXA_APRENDIZADO*
                                                                                                                    this->camadaEscondida[i].neuronios[j].erro*
                                                                                                                    this->camadaEscondida[i-1].neuronios[k].saida*
                                                                                                                    Derivada(this->camadaEscondida[i].neuronios[j].saida));
               }
           }
       }
       /// Atualizando pesos camada sai com ultima escondida ////////////////////////
       for(j=0; j<this->camadaSaida.quantidadeNeuronios; j++)
       {
           for(k=0; k<this->camadaEscondida[this->quantidadeEscondidas-1].quantidadeNeuronios; k++)
           {
               this->camadaSaida.neuronios[j].peso[k] = this->camadaSaida.neuronios[j].peso[k] + (TAXA_APRENDIZADO*
                                                                                                  this->camadaSaida.neuronios[j].erro*
                                                                                                  this->camadaEscondida[this->quantidadeEscondidas-1].neuronios[k].saida*
                                                                                                  Derivada(this->camadaSaida.neuronios[j].saida));
           }
       }
   }
   //cout << "Peso: " << this->camadaEscondida[0].neuronios[0].peso[0] << endl << endl;
   return 0;
}

