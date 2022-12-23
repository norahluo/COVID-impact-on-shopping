#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.stats import sem, t
np.random.seed(310)

class kmeans_missing(object):
    """
    
    Implementation of kmeans clustering considering records with missing values
    
    """
    def __init__(self, potential_centroids, n_clusters, weight):
        
        """
        
        potential_centroids : dataframe
            the records that are used for initialization of centroids
            
        n_clusters : int
            the number of clusters
            
        weight: np.array
            A vector of weight assigned to each feature
            A higher the weight means points in the same group is more alike on that feature than on the others
        
        """
        
        #initialize with potential centroids
        self.weight = weight
        self.n_clusters = n_clusters
        self.potential_centroids = potential_centroids.to_numpy()
        
    def fit(self, data, max_iter=10000, number_of_runs = 1, init = 'random'):
        
        data = data.to_numpy()
        dist_mat = np.zeros((data.shape[0], self.n_clusters))
        all_centroids = np.zeros((self.n_clusters, data.shape[1], number_of_runs))
        costs = np.zeros((number_of_runs,))
        labels = np.zeros((number_of_runs, data.shape[0]))
       

        #####################################################################################################
        ####################################### Initialization Method #######################################
        np.random.seed(310)
        for k in range(number_of_runs):
            
            if init == 'random':
                idx = np.random.choice(range(self.potential_centroids.shape[0]), size = self.n_clusters, replace=False)
                centroids = potential_centroids[idx]
                
            elif init == 'kmeans++':
                idx = []
                candidate = self.potential_centroids
                centroids = np.zeros((self.n_clusters, candidate.shape[1])) 
                
                idx_ = np.random.choice(range(candidate.shape[0])) # initialize the first centroid randomly
                centroids[0] = candidate[idx_]
                candidate = np.delete(candidate, idx_, 0) 
                idx.append(idx_)
                dist_ = np.zeros((candidate.shape[0], len(centroids)-1)) 
               
                for j in range(1, self.n_clusters):  # find the other centroids                  
                    
                    dist_[:,j-1] = np.dot((candidate - centroids[j-1])**2, self.weight) # calculate the distance of candidate to each centroid
                    min_dist = np.min(dist_[:,:j], axis = 1)  # distance between point and the nearest centroid
                    prob = min_dist/sum(min_dist) # probability of a centroid to be chosen is proportional to its distance to the nearest centroid selected
                    idx_ = np.random.choice(range(len(prob)), p = prob)
                    idx.append(idx_)
                    # update centroids and delete it from candidate
                    centroids[j] = candidate[idx_]
                    candidate = np.delete(candidate, idx_, 0)
                    dist_ = np.delete(dist_, idx_, 0)
            

         ################################################################################################### 
         ###################################################################################################
            
            clusters = np.zeros(data.shape[0])
            old_clusters = np.zeros(data.shape[0])
            
            
            for i in range(max_iter):
                
                # Step 1: calculate distance to centroids
                for j in range(self.n_clusters):
                    dist_mat[:,j] = np.nansum(((centroids[j]-data)**2)*self.weight, axis = 1) # for records with nan (missing values), the distance will be calculated using only features with valid value
                    
                # Step 2: Assign to clusters
                old_clusters = clusters
                
                clusters = np.argmin(dist_mat, axis = 1)
                # If a record contains only nan's and 0's, assign it to the group that is closest to the origin
                smallest_cluster_idx = np.argmin(np.sum(centroids, axis = 1)) 
                clusters[np.where(np.nansum(dist_mat, axis = 1) == 0)] = smallest_cluster_idx
                
                # Step 3: Update clusters centroids
                for j in range(self.n_clusters):
                    centroids[j] = np.nanmean(data[clusters == j], axis = 0)
                
                # When # of identified clusters < n_clusters, reset centroids
                
                if np.isnan(centroids).any():
                    # print(self.potential_centroids)
                    # print(idx)
                    centorids = self.potential_centroids[idx]
                
                # the clustering result is the same between two iterations, the algorithm converges
                if all(np.equal(clusters, old_clusters)):
                    break
                
                # Maximum iteration reached before convergence
                if i == max_iter - 1:
                    print('no convergence before maximum iteration!')
                    
                    # Avoid the case that put all records in one cluster
                    # centroids = self.potential_centroids[idx]
                    # for j in range(self.n_clusters):
                        # dist_mat[:,j] = np.nansum(((centroids[j]-data)**2)*self.weight, axis = 1)

            # Record the clustering result for run k
            all_centroids[:,:,k] = centroids # cluster centroids 
            costs[k] = np.mean(np.min(dist_mat, axis = 1)) # the inertia of the clustering result equals the sum of all within group distance
            labels[k] = np.argmin(dist_mat, axis = 1) # group assignment result   
            # If a record contains only nan's and 0's, assign it to the group that is closest to the origin
            smallest_cluster_idx = np.argmin(np.sum(centroids, axis = 1))
            labels[k, np.where(np.nansum(dist_mat, axis = 1) == 0)] = smallest_cluster_idx
        
        # Keep the best clustering result
        self.costs = np.min(costs)
        self.centroids = all_centroids[:,:, np.argmin(costs)]
        self.labels = labels[np.argmin(costs)]
    
    def silhouette(self, data):
        """
        
        Calculate the silhouette score for different number of clusters
        
        s(o) = (b(o) - a(o)) / max(a(o), b(o))
        
        s(o) - silhouette score for point o
        a(o) - the average distance between o and all the other data points in the cluster to which o belongs
        b(o) - the average distance between o to all data points in the nearest cluster to which o does not belong
        
        """
        
        data = data.to_numpy()
        dist_mat = np.zeros((data.shape[0], self.n_clusters))
        sil_ = []
        for k in range(self.n_clusters):
            dist_mat[:,k] = np.nansum((data - self.centroids[k])**2, axis = 1)
        center = np.argmin(dist_mat, axis = 1)
        nearest_center = np.argsort(dist_mat, axis = 1)[:,1]
        
        # calculate the silouette score for each sample
        for i in range(len(data)):
            a = np.mean([np.nansum((data[i] - data[j])**2)**0.5  for j in range(len(center)) if (center[j]==center[i]) & (i != j)])
            b = np.mean([np.nansum((data[i] - data[j])**2)**0.5  for j in range(len(center)) if (center[j]==nearest_center[i])])
            sil_.append((b - a)/max(a,b))
            
        return np.nanmean(sil_) ## use nanmean because there're record with 8 NaNs' in the vector and thus have silhouette score as NaN


def Past_Label(row, kid, centroid):
    """
    This function identifies the pre-pandemic shopping style for each household by comparing the number of online orders they made 
    per week across eight commodity types pre-pandemic to the average number of online orders across eight commodity types of each 
    shopping style group. Euclidean distance is used to calculate the distance between household record and group centroid. 
    
    Input
    --------------
    row : pd.series 
       the number of online orders a household made per week across eight commodity types
       
    kid : pd.series
        number of kids the household has pre-pandemic 
        
    centroid : dataframe, shape(5, 8)
        the average number of online orders made per week across eight commodity types for five shopping style groups 
    
   Return
   ---------------
   Shopping style label : str
   
    """
    
    essential = ['PreparedFood', 'Groceries', 'Clothing','PaperCleaning', 'Medication']
    cat = ['ChildcareItems', 'Clothing', 'HomeOffice', 'Medication',  'PreparedFood', 'OtherFood', 'PaperCleaning', 'Groceries' ]
    
    
    if kid.loc[row.name] > 0:  # If the household has kids pre-pandemic, include childcare items will be included when calculating the distance        
        if sum(row[[cat_ for cat_ in essential+['ChildcareItems']]].isin([0])) > 0: # exclude the household from ECommerce dependent group if never shopped for essential items online            
            if sum(row[['Groceries']].isin([0])) >0: # exclude the household from Partially ECommerce if never shopped for grocery online
                if sum(row[['PreparedFood']].isin([0])) >0: # exclude the household from ENonfood & EPrepFood if never shopped for prepared food online
                    # Calculate group distance and return the label with the min distance
                    return np.sum((row - centroid[:2])**2, axis = 1).idxmin()    
                else:
                    return np.sum((row - centroid[:3])**2, axis = 1).idxmin()
            
            else:
                return np.sum((row - centroid[:4])**2, axis = 1).idxmin()
        else:
            return np.sum((row - centroid)**2, axis = 1).idxmin()
            
    else:
        if sum(row[[cat_ for cat_ in essential]].isin([0])) > 0:
            if sum(row[['Groceries']].isin([0])) > 0:
                if sum(row[['PreparedFood']].isin([0])) > 0:
                    
                    return np.sum((row[cat[1:]] - centroid[:2][cat[1:]])**2, axis = 1).idxmin()
                else:
                    return np.sum((row[cat[1:]] - centroid[:3][cat[1:]])**2, axis = 1).idxmin()
                
            else:
                return np.sum((row[cat[1:]] - centroid[:4][cat[1:]])**2, axis = 1).idxmin()

        else:
            return np.sum((row[cat[1:]] - centroid[cat[1:]])**2, axis = 1).idxmin()
    

        
def independent_ttest(data1, data2, alpha):
    # calculate means
    mean1, mean2 = np.mean(data1), np.mean(data2)
    # calculate standard errors
    se1, se2 = sem(data1), sem(data2)
    # standard error on the difference between the samples
    sed = (se1**2.0 + se2**2.0)**0.5
    # calculate the t statistic
    t_stat = (mean1 - mean2) / sed
    # degrees of freedom
    df = len(data1) + len(data2) - 2
    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) # one tail test
    # return everything
    return t_stat, df, cv, p