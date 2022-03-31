#!/usr/bin/env python
# coding: utf-8

import numpy as np

class kmeans_missing(object):
    def __init__(self, potential_centroids, n_clusters, weight = np.array([1, 3.3, 2.81, 1.79, 5.02, 1.65, 1.65, 37.29])):
        # potential_centroids: the records that are used for initialization of centroids
        # n_clusters: the number of clusters
        # weight: Category with higher shopping rate will get higher weights. A higher the weight means more emphasis on the within-gourp similarity of shopping channel for that category.
        
        #initialize with potential centroids
        self.weight = weight
        self.n_clusters = n_clusters
        self.potential_centroids = potential_centroids.to_numpy()
        
    def fit(self, data, max_iter=10, number_of_runs = 1, init = 'random'):
        
        data = data.to_numpy()
        n_clusters = self.n_clusters
        potential_centroids = self.potential_centroids
        dist_mat = np.zeros((data.shape[0], n_clusters))
        all_centroids = np.zeros((n_clusters, data.shape[1], number_of_runs))
        costs = np.zeros((number_of_runs,))
        labels = np.zeros((number_of_runs, data.shape[0]))
        weight = self.weight
       
        for k in range(number_of_runs):
            
        #####################################################################################################
        ####################################### Initialization Method #######################################
            if init == 'random':
                idx = np.random.choice(range(potential_centroids.shape[0]), size = (n_clusters), replace=False)
                centroids = potential_centroids[idx]
                
            elif init == 'kmeans++':
                candidate = potential_centroids
                centroids = np.zeros((n_clusters, candidate.shape[1])) # shape(k, 8)
                # initialize the first centroid randomly
                idx = np.random.choice(range(candidate.shape[0]))
                centroids[0] = candidate[idx]
                # once selected, no longer be candidate for centroids
                candidate = np.delete(candidate, idx, 0)
               
                dist_ = np.zeros((candidate.shape[0], len(centroids)-1)) # 
                # find the other centroids
                for j in range(1, n_clusters):                   
                    # calculate the distance of candidate to each centroid
                    
                    #dist_[:,j-1] = np.sum((candidate - centroids[j-1])**2, axis = 1)
                    
                    dist_[:,j-1] = np.dot((candidate - centroids[j-1])**2, weight) 
                    
                    # distance between point and the nearest center
                    min_dist = np.min(dist_[:,:j], axis = 1)
                    # probability distribution
                    prob = min_dist/sum(min_dist)
                    idx = np.random.choice(range(len(prob)), size = 1, p = prob)
                    
                    # update centroids and delete it from candidate
                    centroids[j] = candidate[idx]
                    candidate = np.delete(candidate, idx, 0)
                    dist_ = np.delete(dist_, idx, 0)
                    
         ################################################################################################### 
         ###################################################################################################
            
            clusters = np.zeros((data.shape[0],))
            old_clusters = np.zeros(data.shape[0])
            
            
            for i in range(max_iter):
                # Step 1: calculate distance to centroids
                for j in range(n_clusters):
                    # for records with nan, the distance will be calculated using only features with valid value.
                    #dist_mat[:,j] = np.nansum((centroids[j]-data)**2, axis = 1)
                    
                    dist_mat[:,j] = np.nansum(((centroids[j]-data)**2)*weight, axis = 1)
                    
                # Step 2: Assign to clusters
                clusters = np.argmin(dist_mat, axis = 1)
                # If a record contains only nan's and 0's, assign it to the group that is closest to the origin
                smallest_cluster_idx = np.argmin(np.sum(centroids, axis = 1))
                clusters[np.where(np.nansum(dist_mat, axis = 1) == 0)] = smallest_cluster_idx
                
                # Step 3: Update clusters centroids
                for j in range(n_clusters):
                    centroids[j] = np.nanmean(data[clusters == j], axis = 0)
                
                # When # of identified clusters < n_clusters, reset centroids
                if np.isnan(centroids).any():
                    centorids = potential_centroids[idx]
                    
                if all(np.equal(clusters, old_clusters)):
                    break
                    
                if i == max_iter - 1:
                    print('no convergence before maximum iteration!')
                    # Avoid the case that put all records in one cluster
                    centroids = potential_centroids[idx]
                    for j in range(n_clusters):
                        #dist_mat[:,j] = np.nansum((data - centroids[j])**2, axis = 1)
                        dist_mat[:,j] = np.nansum(((centroids[j]-data)**2)*weight, axis = 1)
                else:
                    clusters, old_clusters = old_clusters, clusters # seems not necessary to assign old_clusters to clusters
            
            all_centroids[:,:,k] = centroids
            costs[k] = np.mean(np.min(dist_mat, axis = 1))
            labels[k] = np.argmin(dist_mat, axis = 1)
            smallest_cluster_idx = np.argmin(np.sum(centroids, axis = 1))
            labels[k, np.where(np.nansum(dist_mat, axis = 1) == 0)] = smallest_cluster_idx
            
            
        self.costs = costs
        self.costs = np.min(costs)
        self.best_model = np.argmin(costs)
        self.centroids = all_centroids[:,:, self.best_model]
        self.all_centroids = all_centroids
        self.labels = labels[self.best_model]
    
    def silhouette(self, data):
        data = data.to_numpy()
        n_clusters = self.n_clusters
        centroids = self.centroids
        dist_mat = np.zeros((data.shape[0], n_clusters))
        sil_ = []
        for k in range(n_clusters):
            dist_mat[:,k] = np.nansum((data - centroids[k])**2, axis = 1)
        center = np.argmin(dist_mat, axis = 1)
        nearest_center = np.argsort(dist_mat, axis = 1)[:,1]
        for i in range(len(data)):
            a = np.mean([np.nansum((data[i] - data[j])**2)**0.5  for j in range(len(center)) if (center[j]==center[i]) & (i != j)])
            b = np.mean([np.nansum((data[i] - data[j])**2)**0.5  for j in range(len(center)) if (center[j]==nearest_center[i])])
            sil_.append((b - a)/max(a,b))
            
        return np.nanmean(sil_) ## use nanmean because there're record with 8 NaNs' in the vector and thus have silhouette score as NaN







