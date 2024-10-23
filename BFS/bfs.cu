#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#define MAX_THREADS_PER_BLOCK 256

int no_of_nodes;
int edge_list_size;
FILE *fp;

// Structure to hold a node information
struct Node
{
    int starting;
    int no_of_edges;
};

// Kernel function prototype
__global__ void Kernel(Node* d_graph_nodes, int* d_graph_edges, bool* d_graph_mask, bool* d_graph_visited, int* d_cost, bool* d_over, int no_of_nodes);

// Function declaration
void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
    no_of_nodes = 0;
    edge_list_size = 0;
    BFSGraph(argc, argv);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char** argv) 
{
    printf("Reading File\n");
    static char *input_file_name;

    if (argc == 2) {
        input_file_name = argv[1];
        printf("Input file: %s\n", input_file_name);
    }
    else 
    {
        input_file_name = "SampleGraph.txt";
        printf("No input file specified, defaulting to SampleGraph.txt\n");
    }
    
    // Read in Graph from a file
    fp = fopen(input_file_name, "r");
    if (!fp)
    {
        printf("Error Reading graph file\n");
        return;
    }

    int source = 0;
    
    fscanf(fp, "%d", &no_of_nodes);
    
    int num_of_blocks = 1;
    int num_of_threads_per_block = no_of_nodes;

    // Make execution parameters according to the number of nodes
    if (no_of_nodes > MAX_THREADS_PER_BLOCK)
    {
        num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK); 
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
    }
        
    // Allocate host memory
    Node* h_graph_nodes = (Node*)malloc(sizeof(Node) * no_of_nodes);
    bool *h_graph_mask = (bool*)malloc(sizeof(bool) * no_of_nodes);
    bool *h_graph_visited = (bool*)malloc(sizeof(bool) * no_of_nodes);
    
    int start, edgeno;   
    // Initialize the memory
    for (unsigned int i = 0; i < no_of_nodes; i++) 
    {
        fscanf(fp, "%d %d", &start, &edgeno);
        h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i] = false;
        h_graph_visited[i] = false;
    }
    
    // Read the source node from the file
    fscanf(fp, "%d", &source);
    
    // Set the source node as true in the mask
    h_graph_mask[source] = true;
    
    fscanf(fp, "%d", &edge_list_size);
    
    int id, cost;
    int* h_graph_edges = (int*)malloc(sizeof(int) * edge_list_size);
    for (int i = 0; i < edge_list_size; i++)
    {
        fscanf(fp, "%d", &id);
        fscanf(fp, "%d", &cost);
        h_graph_edges[i] = id;
    }
     
    if (fp)
        fclose(fp);    
    
    printf("Read File\n");

    // Copy the Node list to device memory
    Node* d_graph_nodes;
    cudaMalloc((void**)&d_graph_nodes, sizeof(Node) * no_of_nodes);
    cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(Node) * no_of_nodes, cudaMemcpyHostToDevice);

    // Copy the Edge List to device Memory
    int* d_graph_edges;
    cudaMalloc((void**)&d_graph_edges, sizeof(int) * edge_list_size);
    cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size, cudaMemcpyHostToDevice);
    
    // Copy the Mask to device memory
    bool* d_graph_mask;
    cudaMalloc((void**)&d_graph_mask, sizeof(bool) * no_of_nodes);
    cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice);
    
    // Copy the Visited nodes array to device memory
    bool* d_graph_visited;
    cudaMalloc((void**)&d_graph_visited, sizeof(bool) * no_of_nodes);
    cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice);
    
    // Allocate memory for the result on host side
    int* h_cost = (int*)malloc(sizeof(int) * no_of_nodes);
    for (int i = 0; i < no_of_nodes; i++)
        h_cost[i] = -1;
    h_cost[source] = 0;

    // Allocate device memory for result
    int* d_cost;
    cudaMalloc((void**)&d_cost, sizeof(int) * no_of_nodes);
    cudaMemcpy(d_cost, h_cost, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);

    // Make a bool to check if the execution is over
    bool *d_over;
    cudaMalloc((void**)&d_over, sizeof(bool));

    printf("Copied Everything to GPU memory\n");
    
    // Setup execution parameters
    dim3 grid(num_of_blocks, 1, 1);
    dim3 threads(num_of_threads_per_block, 1, 1);

    // Start the timer
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    float timer_acc = 0.0f;
    int k = 0;
    
    bool stop;
    // Call the Kernel until all the elements of Frontier are not false
    do
    {
        // If no thread changes this value then the loop stops
        stop = false;
        cudaMemcpy(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice);
        
        cudaEventRecord(start_event);
        Kernel<<<grid, threads>>>(d_graph_nodes, d_graph_edges, d_graph_mask, d_graph_visited, d_cost, d_over, no_of_nodes);
        cudaDeviceSynchronize();
        cudaEventRecord(stop_event);
        
        cudaEventSynchronize(stop_event);
        
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
        timer_acc += elapsed_time;

        // Check if kernel execution generated an error
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(error));
            break;
        }
        
        cudaMemcpy(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost);
        k++;
    }
    while (stop);
    
    printf("Kernel Executed %d times\n", k);

    // Copy result from device to host
    cudaMemcpy(h_cost, d_cost, sizeof(int) * no_of_nodes, cudaMemcpyDeviceToHost);

    // Stop the timer
    printf("Processing time: %f (ms)\n", timer_acc);
    
    // Store the result into a file
    FILE *fpo = fopen("result.txt", "w");
    for (int i = 0; i < no_of_nodes; i++)
        fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
    fclose(fpo);
    printf("Result stored in result.txt\n");
    
    // Cleanup memory
    free(h_graph_nodes);
    free(h_graph_edges);
    free(h_graph_mask);
    free(h_graph_visited);
    free(h_cost);
    cudaFree(d_graph_nodes);
    cudaFree(d_graph_edges);
    cudaFree(d_graph_mask);
    cudaFree(d_graph_visited);
    cudaFree(d_cost);
    cudaFree(d_over);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}
