#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <time.h> // 使用 time.h 来获取更精确的时间
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
    const int N = 1;
    double startTime = MPI_Wtime();
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        loadTrainingData("./iris_train.csv");
    }
    
    /* ---------- MPI 数据广播 ---------- */
    /* 1. 先广播训练样本数量 */
    MPI_Bcast(&trainingSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* 2. 准备并广播特征与标签 */
    double featureBuf[MAX_SAMPLES * 4];
    char   labelBuf[MAX_SAMPLES][32];

    if (rank == 0) {
        for (int i = 0; i < trainingSize; ++i) {
            for (int j = 0; j < 4; ++j)
                featureBuf[i * 4 + j] = trainingSet[i].features[j];
            strncpy(labelBuf[i], trainingSet[i].label, 32);
        }
    }
    MPI_Bcast(featureBuf, trainingSize * 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(labelBuf,  trainingSize * 32, MPI_CHAR,   0, MPI_COMM_WORLD);

    double endTime1 = MPI_Wtime();

    /* 输出总时间 */
    if (rank == 0) {
        printf("Total board train set time : %.6f seconds\n", endTime1 - startTime);
    }

    /* 3. 非主进程据此重建训练集 */
    if (rank != 0) {
        for (int i = 0; i < trainingSize; ++i) {
            for (int j = 0; j < 4; ++j)
                trainingSet[i].features[j] = featureBuf[i * 4 + j];
            strncpy(trainingSet[i].label, labelBuf[i], 32);
        }
    }

    double endTime2 = MPI_Wtime();

    /* 输出总时间 */
    if (rank == 0) {
        printf("Total rebuild train set time : %.6f seconds\n", endTime2 - endTime1);
    }

    /* ---------- 读取并广播测试样本 ---------- */
    
    double testSample[4];
    if (rank == 0) {
        FILE *testFile = fopen("./iris_test.txt", "r");
        if (!testFile)
        {
            fprintf(stderr, "Failed to open test file.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 读取测试样本
        if (fscanf(testFile, "%lf %lf %lf %lf",
                &testSample[0], &testSample[1],
                &testSample[2], &testSample[3]) != 4) {
            fprintf(stderr, "Invalid input format\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 打印读取到的测试样本
        printf("Read test sample: %.2f %.2f %.2f %.2f\n",
               testSample[0], testSample[1], testSample[2], testSample[3]);

        fclose(testFile);
    }
    MPI_Bcast(testSample, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double endTime3 = MPI_Wtime();

    /* 输出总时间 */
    if (rank == 0) {
        printf("Total board test set time : %.6f seconds\n", endTime3 - endTime2);
    }

    for (int run = 0; run < N; run++) {
        /* ---------- 各进程计算局部 K 个最近邻 ---------- */
        Neighbor localNeighbors[K];
        for (int i = 0; i < K; ++i) {
            localNeighbors[i].distance = 1e30;   /* 正无穷近似 */
            localNeighbors[i].label[0] = '\0';
        }

        for (int idx = rank; idx < trainingSize; idx += size) {
            double dist = computeDistance(testSample, trainingSet[idx].features);

            /* 找到当前局部 worst（最大距离）的下标 */
            int worst = 0;
            for (int i = 1; i < K; ++i)
                if (localNeighbors[i].distance > localNeighbors[worst].distance)
                    worst = i;

            if (dist < localNeighbors[worst].distance) {
                localNeighbors[worst].distance = dist;
                strncpy(localNeighbors[worst].label, trainingSet[idx].label, 32);
            }
        }
        printf("Current rank is %d , check its K neighbors \n", rank);
        for(int i = 0; i < K; i ++) {
            int cntSetosa = 0, cntVersicolor = 0, cntVirginica = 0;
            for (int i = 0; i < K; ++i) {
                if (strstr(localNeighbors[i].label, "setosa"))       ++cntSetosa;
                else if (strstr(localNeighbors[i].label, "versicolor")) ++cntVersicolor;
                else                                               ++cntVirginica;
            }

            const char *pred = "setosa";
            if (cntVersicolor > cntSetosa && cntVersicolor > cntVirginica)
                pred = "versicolor";
            else if (cntVirginica > cntSetosa && cntVirginica > cntVersicolor)
                pred = "virginica";
            printf("cntSetosa is %d, cntVersicolor is %d, cntVirginica is %d\n", cntSetosa, cntVersicolor, cntVirginica);
        }

        /* ---------- 收集所有进程的最近邻到主进程 ---------- */
        Neighbor *allNeighbors = NULL;
        if (rank == 0)
            allNeighbors = (Neighbor*)malloc(sizeof(Neighbor) * K * size);

        MPI_Gather(localNeighbors, K * sizeof(Neighbor), MPI_BYTE,
                allNeighbors,    K * sizeof(Neighbor), MPI_BYTE,
                0, MPI_COMM_WORLD);

        /* ---------- 主进程做最终 KNN 投票 ---------- */
        if (rank == 0) {
            /* 先按距离整体排序，再取前 K */
            qsort(allNeighbors, K * size, sizeof(Neighbor), compareNeighbors);

            int cntSetosa = 0, cntVersicolor = 0, cntVirginica = 0;
            for (int i = 0; i < K; ++i) {
                if (strstr(allNeighbors[i].label, "setosa"))       ++cntSetosa;
                else if (strstr(allNeighbors[i].label, "versicolor")) ++cntVersicolor;
                else                                               ++cntVirginica;
            }

            const char *pred = "setosa";
            if (cntVersicolor > cntSetosa && cntVersicolor > cntVirginica)
                pred = "versicolor";
            else if (cntVirginica > cntSetosa && cntVirginica > cntVersicolor)
                pred = "virginica";

            if (run == 0) printf("%s\n", pred);
            free(allNeighbors);
        }
    }

    /* 结束计时 */
    double endTime = MPI_Wtime();

    /* 输出总时间 */
    if (rank == 0) {
        printf("Total time for %d runs: %.6f seconds\n", N, endTime - startTime);
    }

    MPI_Finalize();
    return 0;
}