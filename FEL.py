#coding=utf-8
import numpy as np 
import pandas as pd 
from pylab import *
import plotly
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import plot as plot_
import os
import argparse
import warnings 
warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt
plt.switch_backend("Agg")



def FEL_pdf_2D(data,bins=47,cmap=None,KbT=-2.479,fig=None,xlabel_='x',ylabel_='y'):
    """

    """
    #data is a 2D array
    if fig == None:
        fig = '2D_pdf_FEL'
    data = np.array(data)
    print(data.shape)
    if len(data.shape) == 3 and data.shape[1] != 2:
        raise ValueError('FEL data should be a or a serise of 2D array-like and shape should be (n,2) or (m,n,2), e.t. [[1,2],[1,3],....] or [[[1,2],[2,2]...],[[2,1],[1,1]...]... ]')
    elif len(data.shape) == 2 and data.shape[0] != 2:
        raise ValueError('FEL data should be a or a serise of 2D array-like and shape should be (n,2) or (m,n,2), e.t. [[1,2],[1,3],....] or [[[1,2],[2,2]...],[[2,1],[1,1]...]... ]')
    elif len(data.shape) == 2:
        data = [data]
    else:
        raise ValueError('FEL data should be a or a serise of 2D array-like and shape should be (n,2) or (m,n,2), e.t. [[1,2],[1,3],....] or [[[1,2],[2,2]...],[[2,1],[1,1]...]... ]')
    n_figs = len(data)
    w,h = 1,1
    if n_figs > 1:
        w = 2
        h = int(ceil(n_figs/2.0))
    if cmap==None:
        cmap = cm.jet
    figure(figsize=(8*w,6*h))
    for index,d in enumerate(data):
        subplot(h,w,index+1)
        print(d.shape)
        z,xedge, yedge = np.histogram2d(d[0,:], d[1,:], bins=bins)

        x = 0.5*(xedge[:-1] + xedge[1:])
        y = 0.5*(yedge[:-1] + yedge[1:])
        zmin_nonzero = np.min(z[np.where(z > 0)])
        z = np.maximum(z, zmin_nonzero)
        F = KbT*np.log(z)
        F -= np.max(F)
        F = np.minimum(F, 0)
        extent = [yedge[0], yedge[-1], xedge[0], xedge[-1]]
        
        contourf(x,y,F.T,15, cmap=cmap, extent=extent,levels=[i for i in range(-15,0,1)]+[0])
        clb = colorbar()
        clb.set_label('Free energy (kJ/mol)',fontsize=30)
        clb.set_ticks([i for i in range(-20,1,1)][::-1])
        xlabel("Eigenvetor 1",fontsize=30)
        ylabel("Eigenvetor 2",fontsize=30)
        xlim(-12.5,12.5)
        ylim(-12.5,12.5)
         
        plt.xticks([-12,-8,-4,0,4,8,12],fontsize=25)
        plt.yticks([-12,-8,-4,0,4,8,12],fontsize=25)
        #tick_spacing = 2
         
    savefig('{}.eps'.format(fig),dip=2000,bbox_inches='tight')
    print('#'*100)
    print('figure saved to {}.tif!'.format(fig))
    print('#'*100)
    return (x,y,F)

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        # C = map(np.uint8, np.array(cmap(k*h)[:3])*255)
        C = [np.uint8(x) for x in  np.array(cmap(k*h)[:3])*255]
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale
    
def FEL_3D(data,cmap=None,xlabel_='x',ylabel_='y',fig='hills_FEL_3D'):
    if cmap==None:
        cmap = cm.jet
    figure(figsize=(7,6))
    xi,yi,zi = data
    cmap_new = matplotlib_to_plotly(cmap, 255)
    data = [
        go.Surface(x=xi,y=yi,z=zi,
                   colorscale=cmap_new,
                   colorbar=go.surface.ColorBar(title='Free energy (KJ/mol)',titleside='right',titlefont={'size':17}),
                   #lighting=dict(ambient=0.4, diffuse=0.5, roughness = 0.9, specular=0.6, fresnel=0.2),
                   lightposition = dict(x=10,y=-10,z=-200)
                   #contours=go.surface.Contours(z=go.surface.contours.Z(show=True,usecolormap=True,highlightcolor="#42f462",project=dict(z=True)))
        ),

        
    ]
    layout = go.Layout(
        title='Energy landscape',
        titlefont={'size':19},
        autosize=True,
        width=1000,
        height=1000,
        scene={
            'xaxis':{'title': 'CV1','titlefont':{'size':17},'tickfont':{'size':15}},
            'yaxis':{'title': 'CV2','titlefont':{'size':17},'tickfont':{'size':15}},
            'zaxis':{'title': '','tickfont':{'size':15}},
            },
    )
    figs = go.Figure(data=data, layout=layout)
    plot_(figs, filename='{}_3D'.format(fig),auto_open=False)

def read_data(datas):
    data = []
    for d in datas:
        data.append(pd.read_csv(d,comment='#',sep='[,\t ]+',engine='python',header=None)) 

    new_data = []
    for d in data:
        new_data.append(d.iloc[:,-1]) 
    return np.array(new_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pdb",default=None,help = '')
    group = parser.add_mutually_exclusive_group()
    group .add_argument("-pdf",help='Construct free energy landscape by probability density functions.',action='store_true')
    parser.add_argument("-FEL_3d",help='2D free energy landscape is displayed by 3D vision.',action='store_true')
    parser.add_argument("-data",default=None,help='This parameter is the first coordinate value when constructing free energy landscape by probability density function.')
    parser.add_argument("-data1",default=None,help = 'One-dimensional free energy is constructed when only data is specified, and two-dimensional free energy is constructed when data1 is specified at the same time using the probability density function.')
    parser.add_argument("-fig",default='FEL',help = 'The path of saved file of the free energy landscape.')
    parser.add_argument("-kbt",default=-2.479,help = 'Boltzmann constant, default value is 2.479.')
    parser.add_argument("-xy",default=None,nargs='+',help = 'xlabel/ylabel，e.t. "x y"')
    args = parser.parse_args()

    if args.pdf:
        data = [args.data]
        if args.data1:
            data.append(args.data1)
        data = read_data(data)
        if args.xy:
            x,y = args.xy
        else:
            x,y = 'x','y'    
        if data.shape[0] == 2:
            cmap = cm.colors.LinearSegmentedColormap.from_list('new_map',[cm.nipy_spectral(i) for i in range(0,256,1)]+[('white')],15)
            # cmap = cm.colors.LinearSegmentedColormap.from_list('new_map',[cm.nipy_spectral(i) for i in range(0,256,1)]+[('white')],15)
            d = FEL_pdf_2D(data,KbT=-abs(args.kbt),fig=args.fig,xlabel_=x,ylabel_=y,cmap=cmap)    
        else:
            d = FEL_pdf_1D(data,KbT=-abs(args.kbt),fig=args.fig,xlabel_=x)    

    if args.FEL_3d:  
        FEL_3D(d,fig=args.fig,xlabel_=x,ylabel_=y)

  
        


        















