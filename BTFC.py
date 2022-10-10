#!/usr/bin/env python
# coding: utf-8

# # Simple Bayesian Term Frequency Classifier

 
import numpy as np

class BTF:
    
    """ when you initialize BTF we are starting with an empty world """
    
    def __init__(self,training_data):
        self.training_data=training_data
        self.remove=""""0123456789,./?~!@#$%^&*()}{[]\+=?
        😂🤔😑😁😍😉♥😒😭😕🤣😆🙄😃😄👊😜😅😫😛😌😊😓👎🏻🤓🤑😎😝🙄😂"""
        self.st_dists=None
        self.class_dists=None
        
        
    def term_freq(self,list_of_terms):  
        
        """ count term frequencies """
        
        tf={}
        for i in list_of_terms:
            if i not in tf:
                tf[i]=1
            else:
                tf[i]+=1
        for i in tf:
            tf[i]/=len(list_of_terms)
            
        {k: v for k, v in sorted(tf.items(), key=lambda item: item[1],reverse=True)}
            
        return tf
    
    def clean_array(self,array):
        
        """ remove undesireable terms """
        
        clean=[]
        for i in array:
            i=str(i)
            i=i.lower()
            for r in self.remove:
                i=i.replace(r,' ')
            clean+=i.split(" ")
        clean=[i for i in clean if i!=' ' and i!='']
        return clean
        
    def single_term_dists(self,x_label,y_label):
        
        """ construct tf distributions for each class """
        
        dists={}
        self.class_dists={}
        labels=set(self.training_data[y_label])
        for i in labels:
            print("Constructing TF Distribution for label:",i)
            arr=self.training_data[self.training_data[y_label]==i][x_label]
            self.class_dists[i]=len(arr)/len(self.training_data)
            arr=self.clean_array(arr)
            tf=self.term_freq(arr)
            dists[i]=tf
        print("Distributions Constructed")
        self.st_dists=dists
        
    
    def bayes_theorem(self,sample):
        
        """first use law of total probability to get prob of single word occuring
        # then multiply those probabilities together to get total prob of sequence of words occuring"""
        """ this is the denominator of Bayes rule """ 
        
        denom=1
        for i in sample:
            word_prob=0
            for k in self.st_dists:
                try:
                    prob=self.st_dists[k][i]
                except:
                    prob=0
                word_prob+=prob*self.class_dists[k]
            if word_prob>0:
                denom*=word_prob
                
        """ if the sequence of words has an essentially zero prob of occurance
        there is not enough data to classify sample """
        if denom==0:
            return 'null'
        
        """ compute the numerator of Bayes rule """
        """ that is the probability of observing the sequence of words in a given class
        times the probability of a sample being drawn from that class (from class distributions)"""
        
        scores={}
        for k in self.st_dists:
            scores[k]={}
            tot=1
            for i in sample:
                try:
                    prob=self.st_dists[k][i]
                except:
                    prob=0
                if prob>0:
                    tot*=prob
            tot=tot*self.class_dists[k]/denom
            scores[k]=tot
        return [key for key in scores if scores[key]==min(scores.values())][0]
        
    def single_term_test(self,test_samples_x,test_samples_y):
        num_correct=0
        num_zeros=0
        labels=list(self.class_dists.keys())
        probs=list(self.class_dists.values())
        for i in range(len(test_samples_x)):
            sample=self.clean_array([test_samples_x[i]])
            classification=self.bayes_theorem(sample)
            if classification=='null':
                num_zeros+=1
                classification=np.random.choice(labels,p=probs)
            if classification==test_samples_y[i]:
                num_correct+=1
        print('score',num_correct/len(test_samples_x))
        print("null",num_zeros)
            
            
        
            
        

