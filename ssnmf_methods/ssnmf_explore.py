import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy.io import loadmat

def plot_utility(x,y,name):
	plt.figure()      
	ax = plt.subplot(111)    
	ax.spines["top"].set_visible(False)    
	ax.spines["bottom"].set_visible(True)    
	ax.spines["right"].set_visible(False)    
	ax.spines["left"].set_visible(True)    

	plt.plot(x,y, "-", lw = 2.5, color = "black")

	ax.get_xaxis().tick_bottom()    
	ax.get_yaxis().tick_left()    
	  
	#plt.ylim(min(errs)*0.9, max(errs)*1.1)    
	plt.xlim(1, numIters)    

	plt.title(name)
	plt.xlabel('Iteration Number')
	#plt.ylabel(name)
	#plt.text(x_ccord, y_coord, "your message", fontsize=10)    

	#plt.yticks(range(start, finish, interval), [str(x) + "%" for x in range(start, finish, interval)], fontsize=14)    
	plt.yticks(fontsize=10)
	plt.xticks(fontsize=10)    


	plt.savefig(name + ".png", bbox_inches="tight") 
	plt.close()


### Kullback-Liebler supervised NMF (KLSNMF)
## This routine minimizes ||X - AS||_F^2 + \lambda*D_{KL}(Y||BS)
# Inputs : X (m by n) np.array flattened image data; Y (p by n) np.array labels; k integer number of topics; kwargs boolean values for saving
# Optional Inputs: A np.array; S np.array; B np.array 
# Outputs: A (m by k); S (k by n); B (p by n)
# Optional Outputs: errs np.array; reconerrs np.array; classerrs np.array

def klsnmfmult(X,Y,k,**kwargs):
    rows, cols = np.shape(X)
    classes, Ycols = np.shape(Y)
    
    # Get optional arguments or set to default
    A = kwargs.get('A', np.random.rand(rows,k))
    S = kwargs.get('S', np.random.rand(k,cols))
    B = kwargs.get('B', np.random.rand(classes,k))
    lam = kwargs.get('lam', 1)
    numiters = kwargs.get('numiters', 10)
    saveerrs = kwargs.get('saveerrs', False)
    
    if(saveerrs):
        errs = np.empty(numiters)      #initialize error array
        reconerrs = np.empty(numiters)
        classerrs = np.empty(numiters)
    
    # Multiplicative updates for A, S, and B
    for i in range(numiters):
        A = np.multiply(np.divide(A,A @ S @ np.transpose(S)), X @ np.transpose(S))
        B = np.multiply(np.divide(B,np.ones((classes,cols)) @ np.transpose(S)), np.divide(Y, B @ S) @ np.transpose(S))
        S = np.multiply(np.divide(S, 2 * np.transpose(A) @ A @ S + lam * np.transpose(B) @ np.ones((classes,cols))),
                       2 * np.transpose(A) @ X + lam * np.transpose(B) @ np.divide(Y, B @ S)) 
        
        # Save errors
        if(saveerrs):
            errs[i] = la.norm(X - A @ S, 'fro') + lam * la.norm(Y - B @ S, 'fro') 
            reconerrs[i] = la.norm(X - A @ S, 'fro')
            classerrs[i] = la.norm(Y - B @ S, 'fro')
        
    if(saveerrs):
        return A, B, S, errs, reconerrs, classerrs
    else:
        return A, B, S

data = loadmat('/media/tmerkh/G_Drive/covid-ssnmf/ssnmf_methods/btstrpCOVID300.mat')
numIters = 20
A,B,S,errs,reconerrs,classerrs = klsnmfmult(data['btstrpimagemat'],data['btstrplabelmat'],10,numiters = numIters, saveerrs = True,lam=100000)

plot_utility(range(1,numIters+1), errs, "Error")

# Things to do:
## 1) Figure out how to interpret the quality of the minimization/result
## 2) Try this out with CT data and so on
## 3) Write things out in an organized fashion so others know what is going on