#pragma once

//#define TAXA_APRENDIZADO    (0.00001)
#define TAXA_APRENDIZADO    (0.00001)
#define BIAS 1



using namespace std;



class Neuronio{
    public:
        int quantidadeLigacoes;
        double *peso;
        double erro;
        double saida;

        Neuronio();
        Neuronio(int);
        //~Neuronio(void);

        /*
        int getQuantidadeLigacoes() const;
        double* getPeso() const;
        void setErro();
        void setSaida();
        */
};

class Camada{
    public:
        int quantidadeNeuronios;
        Neuronio *neuronios;

        Camada();
        Camada(int,int);
        //~Camada(void);


};

class RedeNeural{
    public:
        Camada camadaEntrada;
        Camada *camadaEscondida;
        Camada camadaSaida;

        int quantidadeEscondidas;

        RedeNeural(int,int,int,int);
        //~RedeNeural(void);
        void RNA_copiarParaEntrada(double *);
        void RNA_copiarParaSaida(double *);
        void RNA_calcularSaida(void);
        int RNA_backPropagation(double *, double *);
};

