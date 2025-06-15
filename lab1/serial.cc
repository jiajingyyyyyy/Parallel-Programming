#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define MAX_SAMPLES 200
#define MAX_LINE_LEN 256
#define K 3

typedef struct
{
    double features[4];
    char label[32];
} IrisSample;

typedef struct
{
    double distance;
    char label[32];
} Neighbor;

IrisSample trainingSet[MAX_SAMPLES];
int trainingSize = 0;

int split(const char *line, char tokens[][64], char delimiter)
{
    int count = 0;
    const char *start = line;
    const char *ptr = line;
    while (*ptr)
    {
        if (*ptr == delimiter)
        {
            strncpy(tokens[count], start, ptr - start);
            tokens[count][ptr - start] = '\0';
            count++;
            start = ptr + 1;
        }
        ptr++;
    }
    if (ptr > start)
    {
        strcpy(tokens[count++], start);
    }
    return count;
}

void loadTrainingData(const char *path)
{
    FILE *file = fopen(path, "r");
    if (!file)
    {
        fprintf(stderr, "Failed to open training file.\n");
        return;
    }

    char line[MAX_LINE_LEN];
    int lineNum = 0;
    while (fgets(line, sizeof(line), file))
    {
        lineNum++;
        if (line[strlen(line) - 1] == '\n')
            line[strlen(line) - 1] = '\0';

        char parts[6][64];
        int partCount = split(line, parts, ',');

        if (lineNum == 1 && (strcmp(parts[0], "Sepal.Length") == 0 || strcmp(parts[0], "Species") == 0))
        {
            continue;
        }

        if (partCount < 6)
            continue;

        IrisSample sample;
        for (int i = 0; i < 4; i++)
        {
            sample.features[i] = atof(parts[i + 1]);
        }
        strncpy(sample.label, parts[5], sizeof(sample.label));
        trainingSet[trainingSize++] = sample;
    }

    fclose(file);
}

double computeDistance(const double *a, const double *b)
{
    double sum = 0.0;
    for (int i = 0; i < 4; i++)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

int compareNeighbors(const void *a, const void *b)
{
    Neighbor *na = (Neighbor *)a;
    Neighbor *nb = (Neighbor *)b;
    return (na->distance > nb->distance) - (na->distance < nb->distance);
}

int main(int argc, char **argv)
{
    
    const int N = 10000000;
    // 加载训练数据
    loadTrainingData("./iris_train.csv");

    // 读取测试样本
    double testSample[4];
    FILE *testFile = fopen("./iris_test.txt", "r");
        if (!testFile)
        {
            fprintf(stderr, "Failed to open test file.\n");
        }

        // 读取测试样本
        if (fscanf(testFile, "%lf %lf %lf %lf",
                &testSample[0], &testSample[1],
                &testSample[2], &testSample[3]) != 4) {
            fprintf(stderr, "Invalid input format\n");
        }

        // 打印读取到的测试样本
        printf("Read test sample: %.2f %.2f %.2f %.2f\n",
               testSample[0], testSample[1], testSample[2], testSample[3]);

        fclose(testFile);
    clock_t start = clock();
    for (int run = 0; run < N; ++run)
    {
        // 计算所有训练样本与测试样本的距离
        Neighbor allNeighbors[MAX_SAMPLES];
        
        for (int i = 0; i < trainingSize; i++)
        {
            allNeighbors[i].distance = computeDistance(testSample, trainingSet[i].features);
            strncpy(allNeighbors[i].label, trainingSet[i].label, 32);
        }

        // 按距离排序
        qsort(allNeighbors, trainingSize, sizeof(Neighbor), compareNeighbors);

        // KNN投票
        int cntSetosa = 0, cntVersicolor = 0, cntVirginica = 0;
        for (int i = 0; i < K; ++i)
        {
            if (strstr(allNeighbors[i].label, "setosa"))
                ++cntSetosa;
            else if (strstr(allNeighbors[i].label, "versicolor"))
                ++cntVersicolor;
            else
                ++cntVirginica;
        }

        // 确定预测结果
        const char *pred = "setosa";
        if (cntVersicolor > cntSetosa && cntVersicolor > cntVirginica)
            pred = "versicolor";
        else if (cntVirginica > cntSetosa && cntVirginica > cntVersicolor)
            pred = "virginica";
        if(run == 0) printf("%s\n", pred);
    }
    clock_t end = clock();

    // 计算总时间
    double totalTime = (double)(end - start) / CLOCKS_PER_SEC;

    // 输出总时间
    printf("Total time for %d runs: %.6f seconds\n", N, totalTime);


    return 0;
}
