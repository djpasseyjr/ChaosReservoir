import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
import pickle
import datetime as dt                   # to add the date and time into the figure title
import time
import os
import subprocess
import sys, traceback #needed for when I use jupyter notebook
import glob #could use but don't need to right now

DROP_PARTIAL_EXPERIMENTS = 25 #used in optimization constructor
combine_topos = True
SCALE_DOWN = False
# DIR = '/Users/joeywilkes/ReservoirComputing/research_data'
FILE_LIST = [
    # 'compiled_output_f2_38_barab2.pkl'
#     ,'compiled_output_jw44_barab2.pkl'
    'compiled_output_jw39_barab1.pkl'
    ,'compiled_output_jw43_barab1.pkl'
    ,'compiled_output_jw40_watts3.pkl'
    ,'compiled_output_jw45_watts3.pkl'
#     ,'compiled_output_jj6_random_digraph.pkl'
#     ,'compiled_output_jj7_erdos.pkl'
#     ,'compiled_output_jw53_ident.pkl'
    ,'compiled_output_jw54_loop.pkl'
    # ,'compiled_output_jw55_no_edges.pkl'
#     ,'compiled_output_jw56_chain.pkl'
]

SAVEFIGS = True
RESOLUTION = int(1e2)
DIR = None
FILE_LIST = None
NUM_WINNERS = 100 #find top NUM_WINNERS in grid search optimization

#selection for either 'visualize' or 'optimize' or None (means both)
SELECTION = None
SELECTION = 'visualize'
# SELECTION = 'evaluate'
# SELECTION = 'optimize'
# SELECTION = ['optimize','visualize']
# LOCATION FOR OUTPUT FILES
LOC = None
month, day = dt.datetime.now().month, dt.datetime.now().day
LOC = f'BEST_DATA_AS_OF_{month}_{day}'

DROP_VALUES = {
  'adj_size':[]
  ,'topo_p':[]
  ,'gamma':[]
  ,'sigma':[]
  ,'spect_rad':[]
  ,'ridge_alpha':[]
}

if DIR is None:
    DIR = ''

class Optimize:
    """ """
    def __init__(self,df_dict):
        """
        Parameters:
            df_dict     (dict): keys are topology name strings, and values are the dataframe for
                                that topology, aka output of df_dict function
        """
        # currently DROP_PARTIAL_EXPERIMENTS is a global variable
        if DROP_PARTIAL_EXPERIMENTS:
            for t in df_dict.keys():
                x = df_dict[t]
                x.drop(index=x[x['num_nets_by_exp'] < DROP_PARTIAL_EXPERIMENTS ].index,inplace=True)
        self.data = df_dict
        self.topos = dict()
        self.compare = dict()
        self.best = dict()

    def win(self,num_winners=5,loc=None,save_best=True):
        """
        Get top `num_winners` from each topology for both model types (thinned or not thinned)
        as well as winners out of any topology (comparing) by model type (thinned or not thinned)
        as well as winners out of all model types out of all topologies (the best model from this data)

        Assume "dense" means not thinned in this case, remove_p = 0

        Parameters:
            num_winners     (int): number of winners from each topology to consider
            loc             (str): location/directory for output file
            save_best       (bool): whether to save the best results as a pkl file

        Returns:
            results  (dict): contains the following keys and results
                topos    (dict): dictionary containing dataframes for each topologys best with thinning and no thinning ('dense')
                compare  (dict): compares all the topologies, to see which topologies are the best
                best     (df): merged & sorts the two dataFrames in compare values

        """
        if num_winners <= 0:
            raise ValueError('num_winners should be greater than zero')

        if not loc:
             loc = ''
        else:
            if loc[-1] != '/':
                loc += '/'

        # create dictionaries by topology then by thinned or not thinned / "dense"
        #only include topologies that there is data for
        for i in self.data.keys():
            self.topos[i] = {'thinned':None,'dense':None}
            self.compare = {'thinned':None,'dense':None}

            temp = self.data[i].copy()

            dense = temp[temp.remove_p == 0].copy()
            x = dense.groupby(['exp_num']).aggregate(np.mean).copy()
            x.sort_values(by=['mean_pred','mean_err'],ascending=[False,True],inplace=True)
            self.topos[i]['dense'] = x.iloc[:num_winners]
            # exclude equal to zero to can compare whether a thinned network can beat a non thinned network
            # and to avoid having a not thinned network appear in both sides
            thin = temp[temp.remove_p > 0].copy()
            y = temp.groupby(['exp_num']).aggregate(np.mean).copy()
            y.sort_values(by=['mean_pred','mean_err'],ascending=[False,True],inplace=True)
            self.topos[i]['thinned'] = y.iloc[:num_winners]

        # combine all the dense networks together, and the thinned ones together to see compare topologies for
        # either thinned or dense -
        # then combine the dense & the thinned ones together to get the best
        best = pd.DataFrame()
        for m in ['thinned','dense']:
            df = self.topos[list(self.data.keys())[0]][m].copy()
            if 'net' not in df.columns:
                df['net'] = list(self.data.keys())[0]
            for i in list(self.data.keys())[1:]:
                temp = self.topos[i][m]
                if 'net' not in temp.columns:
                    # temp.loc[:,'net'] = i #net isnt in the column remember
                    temp['net'] = i
                df = df.append(temp,ignore_index=True)
            df.sort_values(by=['mean_pred','mean_err'],ascending=[False,True],inplace=True)
            self.compare[m] = df
            best = best.append(df,ignore_index=True)

        self.best = best

        results = dict()
        results['topos'] = self.topos
        results['compare'] = self.compare
        results['best'] = self.best

        #write the best dataframe to a pickle file
        #because in the super computer returning the results might not be useful
        month, day = dt.datetime.now().month, dt.datetime.now().day
        hour, minute = dt.datetime.now().hour, dt.datetime.now().minute
        file = f'best_as_of_{month}_{day}_at_{hour}_{minute}.pkl'
        #location may be basically empty
        if save_best:
            name = loc + file
            best.to_pickle(name)

        return results

class Visualize:
    """Visualization tool"""

    def __init__(self,df_dict,drop_values=None):
        """
        Initialize data members for visualizations

        Parameters:
            df_dict         (dict): keys are topology name strings, and values are the dataframe for
                                    that topology, aka output of df_dict function
            drop_values     (dict): dict where keys are parameters, and value for each parameter are the values to drop
        """

        #used in compare_parameters (at least)
        self.parameter_names = {'adj_size':'Network Size',
          'topo_p': 'Mean Degree / Rewiring Prob',
          'gamma': 'Gamma',
          'sigma': 'Sigma',
          'spect_rad':'Spectral Radius',
          'ridge_alpha': 'Ridge Alpha (Regularization)'}
        self.topo_names = {
             'barab1':'Barabasi-Albert 1'
             ,'barab2':'Barabasi-Albert 2'
             ,'erdos':'Erdos-Reyni'
             ,'random_digraph':'Random Directed'
             ,'ident':'Identity'
             ,'watts2':'Watts-Strogatz 2'
             ,'watts4':'Watts-Strogatz 4'
             ,'watts3':'Watts-Strogatz 3'
             ,'watts5':'Watts-Strogatz 5' #this said 4, maybe the reason for 2 watts4 graphs
             ,'geom':'Random Geometric'
             ,'no_edges':'No Edges'
             ,'loop':'Loop'
             ,'chain':'Chain'
         }
        self.var_names = {
            'mean_pred':'Avg. Accuracy Dur.'
            ,'mean_err':'Avg. Fit Error'
            ,'remove_p':'Edge Removal %'
            ,'ncd':'# Nets (log)' #net count distribution
        }
        self.legend_size = 10
        # self.figure_width_per_column = 6.5
        self.figure_width_per_column = 9 # when legend is bigger (legend_size = 10)
        self.figure_height_per_row = 4 #or low as 2.5

        # may only want to drop values in visualization, not optimize

        if drop_values is not None:
            for p in drop_values.keys():
                if p not in self.parameter_names.keys():
                    print(p,'not in',self.parameter_names.keys())
                else:
                    for val in drop_values[p]:
                        for topo in df_dict.keys():
                            df = df_dict[topo]
                            if val not in df[p].unique():
                                print(f'{val} is not in {p} for {topo}')
                            else:
                                df.drop(index=df.loc[df[p] == val ].index,inplace=True)
                                df_dict[topo] = df

        self.data = df_dict

    def tolerance(self, **kwargs):
        """A helper function to increase the fault tolerance of the class

        Parameters:
            **kwargs

        Returns:
            results (dict):

        """
        raise NotImplementedError('tolerance not done yet  ')
        network_options = {'barab1', 'barab2',
                        'erdos', 'random_digraph',
                        'watts3', 'watts5',
                        'watts2','watts4',
                        'geom', 'no_edges',
                        'loop', 'chain',
                        'ident'
                      }

        if dir:
            if dir[-1] != '/':
                path = dir + '/'
        else:
            #to simply upcoming code
            path = ''

    def view_parameters(
        self,
        t,
        loc=None,
        dep = 'mean_pred',
        savefig = None,
        res = int(1e2),
        verbose = False
        ):
        """
        Create visualizations viewing parameter values influence on dependent variable
        as well as the distribution of number of nets for each remove_p per value for each
        parameter (as a logarithmic line plot)

        Parameters:
            t       (str): topology
            loc     (str): location / directory for output files
            dep     (str): dependent variable
            savefig (str): location to save figure, if None, then don't save the figure
            res     (int): resolution of images
            verbose (str): debugging purposes
        """
        hyp_p = list(self.parameter_names.keys()) #cast as a list so topo_p could be removed
        # hyp_p = hyp_p[:2] #don't plot as much
        resolution = res
        network_options = ['barab1', 'barab2',
                            'erdos', 'random_digraph',
                            'watts3', 'watts5',
                            'watts2','watts4',
                            'geom', 'no_edges',
                            'loop', 'chain',
                            'ident'
                          ]

        if t not in ['erdos', 'random_digraph','watts3', 'watts5','watts2','watts4','geom']:
            #this throws an error when it's not necessary
            hyp_p.remove('topo_p')

        if not loc:
             loc = ''
        else:
            if loc[-1] != '/':
                loc += '/'

        if t not in network_options:
            raise ValueError(f'{t} not in {network_options}')

        x = self.data[t]

        if dep not in ['mean_pred','mean_err']:
            print(f'{dep} is not acceptable as dependent variable ')

        num_columns = 2

        fig_height = self.figure_height_per_row * len(hyp_p)
        fig_width = self.figure_width_per_column * num_columns
        fig, ax = plt.subplots(len(hyp_p),num_columns,sharey=False,dpi=resolution,figsize=(fig_width,fig_height))
        #     gs = gridspec.GridSpec(1,len(hyp_p),figure=fig,hspace=HS)
        rp = x.remove_p.unique()
        # each subplot is a different hyperparameter
        #for each subplot, plot all unique values of that parameter
        e = x[x.net == t].copy()
        for i,v in enumerate(hyp_p):
            for j,p in enumerate(sorted(x[v].unique())):
                if verbose:
                    print(i,v,j,p,'\n')
                S = e[e[v] == p].groupby(e.remove_p).aggregate(np.mean)[dep].copy()
                ax[i][0].plot(S.index,S.values,label=p) #if one topology
                ax[i][0].scatter(S.index,S.values)

                A = pd.DataFrame(x.loc[(x.net == t) & (x[v] == p)]['remove_p'].value_counts())
                A.reset_index(inplace=True)
                A.sort_values(by='index',inplace=True)
                ax[i][1].semilogy(A['index'],A['remove_p'],label=p)

            leg0 = ax[i][0].legend(prop={'size': self.legend_size},bbox_to_anchor=(-0.1, 0.7)) #bbox_to_anchor=(x,y)
            ax[i][0].set_title(f'{self.parameter_names[v]} Value Comparison')
            ax[i][0].set_xlabel(self.var_names['remove_p'])
            ax[i][0].set_ylabel(self.var_names[dep])

            leg1 = ax[i][1].legend(prop={'size': self.legend_size},bbox_to_anchor=(1.01, 0.7)) ##bbox_to_anchor=(x,y)
            ax[i][1].set_xlabel(self.var_names['remove_p'])
            ax[i][1].set_ylabel(self.var_names['ncd'])
            ax[i][1].set_title(f'{self.parameter_names[v]} value counts per value')

        my_suptitle = fig.suptitle(f'{self.topo_names[t]} Hyper-Parameter Comparison', fontsize=16,y=1.03)
        plt.tight_layout()
        # plt.show()

        if savefig:
            month, day = dt.datetime.now().month, dt.datetime.now().day
            hour, minute = dt.datetime.now().hour, dt.datetime.now().minute
            if dep == 'mean_pred':
                fig.savefig(loc + f'Optimize_{t}_{month}_{day}_at_{hour}_{minute}.png',bbox_inches='tight',bbox_extra_artists=[my_suptitle,leg0,leg1])
            else:
                fig.savefig(loc + f'Optimize_{t}_by_fit_{month}_{day}_at_{hour}_{minute}.png',bbox_inches='tight',bbox_extra_artists=[my_suptitle,leg0,leg1])
        print('finished with ',t,dep)

    def view_dependents():
        """
        View both the mean_pred and mean_error

        """
        raise NotImplementedError('not done')
        # fix legend size
        # parameters for function
        resolution = int(1e2)
        verbose = False
        fig_height = 3 * len(hyp_p)
        fig_width = 13

        #In this case, sharey=True would make the visualization nearly useless
        fig, ax = plt.subplots(len(hyp_p),2,sharey=False,dpi=resolution,figsize=(fig_width,fig_height))

        rp = x.remove_p.unique()
        # each subplot is a different hyperparameter
        #for each subplot, plot all unique values of that parameter
        t = 'watts5'
        # dep for dependendent variable


        e = x[x.net == t].copy()
        for i,v in enumerate(hyp_p):
            for j,p in enumerate(x[v].unique()):
                S = e[e[v] == p].groupby(e.remove_p).aggregate(np.mean)['mean_pred'].copy()
                ax[i][0].plot(S.index,S.values,label=p) #if one topology
                ax[i][0].scatter(S.index,S.values)

                S = e[e[v] == p].groupby(e.remove_p).aggregate(np.mean)['mean_err'].copy()
                ax[i][1].plot(S.index,S.values,label=p) #if one topology
                ax[i][1].scatter(S.index,S.values)


            leg0 = ax[i][0].legend(prop={'size': 8},bbox_to_anchor=(-0.2, 0.5))
            ax[i][0].set_title(f'Mean Predict; {self.parameter_names[v]} Value Comparison')
            ax[i][0].set_xlabel(self.var_names['remove_p'])
            ax[i][0].set_ylabel(self.var_names['mean_pred'])
            leg1 = ax[i][1].legend(prop={'size': 8},bbox_to_anchor=(1.2, 0.5))
            ax[i][1].set_title(f'Mean Fit Error; {self.parameter_names[v]} Value Comparison')
            ax[i][1].set_xlabel(self.var_names['remove_p'])
            ax[i][1].set_ylabel(self.var_names['mean_err'])

        my_suptitle = fig.suptitle(f'{self.topo_names[t]} Hyper-Parameter Comparison', fontsize=16,y=1.01)
        plt.tight_layout()

    def view_topology():
        """view the dependent variables as well as the net count distribution per parameter value

        """
        raise NotImplementedError('not done')
        # with NCD
        #change parameter_names
        print('where is hyp_p initialized')
        # add self.legend_size
        # add save figure
        # add parameters to function (topology, etc)
        # add legend_x and legend_y
        resolution = int(1e2)
        verbose = False
        fig_height = 3 * len(hyp_p)
        fig_width = 25
        legend_size = 10
        legend_x = -0.2
        legend_y = 0.7
        # In this case, sharey=True would make the visualization nearly useless
        fig, ax = plt.subplots(len(hyp_p),3,sharey=False,dpi=resolution,figsize=(fig_width,fig_height))

        rp = x.remove_p.unique()
        # each subplot is a different hyperparameter
        #for each subplot, plot all unique values of that parameter
        t = 'erdos'
        print('t=',t,'\nshould probably be adjusted to not be hard coded ')
        # dep for dependendent variable

        e = x[x.net == t].copy()
        for i,v in enumerate(hyp_p):
            for j,p in enumerate(x[v].unique()):
                S = e[e[v] == p].groupby(e.remove_p).aggregate(np.mean)['mean_pred'].copy()
                ax[i][0].plot(S.index,S.values,label=p) #if one topology
                ax[i][0].scatter(S.index,S.values)

                S = e[e[v] == p].groupby(e.remove_p).aggregate(np.mean)['mean_err'].copy()
                ax[i][1].plot(S.index,S.values,label=p) #if one topology
                ax[i][1].scatter(S.index,S.values)

                A = pd.DataFrame(x.loc[(x.net == t) & (x[v] == p)]['remove_p'].value_counts())
                A.reset_index(inplace=True)
                A.sort_values(by='index',inplace=True)
                ax[i][2].semilogy(A['index'],A['remove_p'],label=p)


            leg0 = ax[i][0].legend(prop={'size': legend_size},bbox_to_anchor=(legend_x, legend_y))
            ax[i][0].set_title(f'Mean Predict; {self.parameter_names[v]} Value Comparison')
            ax[i][0].set_xlabel('Edge Removal %')
            ax[i][0].set_ylabel('Mean Prediction Duration')
            leg1 = ax[i][1].legend(prop={'size': legend_size},bbox_to_anchor=(legend_x, legend_y))
            ax[i][1].set_title(f'Mean Fit Error; {self.parameter_names[v]} Value Comparison')
            ax[i][1].set_xlabel('Edge Removal %')
            ax[i][1].set_ylabel('Mean Fit Error')
            leg2 = ax[i][2].legend(prop={'size': legend_size},bbox_to_anchor=(legend_x, legend_y))
            ax[i][2].set_xlabel('Edge Removal %')
            ax[i][2].set_ylabel('# nets (log)')
            ax[i][2].set_title(f'{self.parameter_names[v]} value counts per value')

        my_suptitle = fig.suptitle(f'{self.topo_names[t]} Comprehensive View', fontsize=16,y=1.02)
        plt.tight_layout()

    def compare_parameter(self,
        parameter,
        loc,
        dep = 'mean_pred',
        savefig = False,
        res = int(1e2),
        verbose = False,
        compare_topos = None
        ):
        """For a given Parameter, compare the topologies
        display all the topologies

        Parameters:
            parameter       (str): string describing the parameter to compare
            dep             (str): dependent variable
            savefig         (bool): whether or not to export the figure
            res             (int): resolution of images
            verbose         (str): debugging purposes
            compare_topos   (list): indicates which topologies to include in the comparison, if None then compare all available in self.data

        """
        # input = dict()
        # input['t'] = t
        # input['dep'] = dep
        # input['savefig'] = savefig
        # input['res'] = res
        # tol_results = self.tolerance(input)
        # dep = tol_results['dep']
        # resolution = tol_results['res']
        # parameter = tol_results['parameter']

        if not loc:
             loc = ''
        else:
            if loc[-1] != '/':
                loc += '/'

        if compare_topos:
            num_unique_nets = len(compare_topos)
        else:
            num_unique_nets = len(self.data.keys())
            compare_topos = self.data.keys()

        num_columns = 1
        fig_height = self.figure_height_per_row * num_unique_nets
        fig_width = self.figure_width_per_column * num_columns

        fig, ax = plt.subplots(num_unique_nets,num_columns,sharey=False,dpi=res,figsize=(fig_width,fig_height))

        #each subplot is a topology
        #then all values for the hyper-parameter are compared

        v = parameter

        for i,t in enumerate(compare_topos):
            e = self.data[t].copy()
            for j,p in enumerate(sorted(e[v].unique())):
                S = e[e[v] == p].groupby(e.remove_p).aggregate(np.mean)[dep].copy()
                ax[i].plot(S.index,S.values,label=p) #if one topology
                ax[i].scatter(S.index,S.values)
            leg0 = ax[i].legend(loc='lower left',prop={'size': self.legend_size},bbox_to_anchor=(1.01, 0.2)) #bbox_to_anchor=(x,y)
            ax[i].set_title(self.topo_names[t])
            ax[i].set_xlabel(self.var_names['remove_p'])
            ax[i].set_ylabel(self.var_names[dep])
        # title_ = v.upper().replace('_',' ')
        # fig.suptitle(f'{title_} Comparison For All Topologies', fontsize=16,y=1.03)
        my_suptitle = fig.suptitle(f'{self.parameter_names[v]} Comparison For All Topologies', fontsize=16,y=1.02)
        plt.tight_layout()
        # plt.show()

        if savefig:
            month, day = dt.datetime.now().month, dt.datetime.now().day
            hour, minute = dt.datetime.now().hour, dt.datetime.now().minute
            if dep == 'mean_pred':
                fig.savefig(loc + f'Compare_{parameter}_{month}_{day}_at_{hour}_{minute}.png',bbox_inches='tight',bbox_extra_artists=[my_suptitle,leg0])
            else:
                fig.savefig(loc + f'Compare_{parameter}_by_fit_{month}_{day}_at_{hour}_{minute}.png',bbox_inches='tight',bbox_extra_artists=[my_suptitle,leg0])

    def contrast_topos(self,savefigs):
        """Plot all the topologies against each other """
        print('how to limit to either a subset of parameters, or best')
        raise NotImplementedError('contrast_topos not done ')

        # use the best network from each topology as given by the optimize class

        O = Optimize(self.data)
        results = O.win(num_winners=1)
        thin = results['compare']['thinned']
        dense = results['compare']['dense']
        # get the parameters for each topology, then we want to see as remove_p varies (note uneven amount of data)


        #4 SUBPLOTS, top left is horizontal bar graph which each topology, compare (best thinned) & dense per topo on one plot
        #top right is distribution of networks
        #bottom left is comparison as remove_p varies
        #bottom right is ncd (net count distribution)


        # to do the bottom left, get the parameters for each topology

        if savefigs:
            raise NotImplementedError('not done')

    def network_statistics(self):
        """ """
        raise NotImplementedError('network_statistics not done ')

    def ncc(self,
        loc,
        savefig = False,
        res = int(1e2),
        verbose = False,
        ):
        """ Generate figures to look at the number of connected components for each remove_p value """
        # each row is a topology, and each column is either nscc or nwcc

        num_columns = 2
        num_topos = len(list(self.data.keys()))
        fig_height = self.figure_height_per_row * num_topos
        fig_width = self.figure_width_per_column * num_columns
        fig, ax = plt.subplots(num_topos,num_columns,sharey=False,dpi=res,figsize=(fig_width,fig_height))


        # for i,t in enumerate(self.data.keys()):
        #     e = self.data[t].copy()
        #     for j,p in enumerate(['nscc','nwcc']):
        #         #use seaborn box & whisker plot with axes?
        #         ax[i][j].

        raise NotImplementedError('ncc not done ')

    def all(self
        ,selection=None
        ,savefigs=True
        ,resolution=int(1e2)
        ,loc=None
        ,verbose=False):
        """ The method will will create all the visuals

        Parameters
            selection  (list / str): which visuals to include, if None then visualize all
            savefigs    (bool): save the figures
            resolution  (int): resolution for the figures
            loc         (str): location / directory for output files
            verbose     (str): verbose output

        """
        options = ['view_parameters','compare_parameter','compare_parameters']
        l = []
        # The user could accidentally think parameter was plural

        if selection is None or selection == 'None':
            l = options
        else:
            if isinstance(selection,list):
                for i in selection:
                    if i not in set(options):
                        raise ValueError(f'{i} must be None or in {options}')
                    else:
                        l.append(i)
            else:
                if selection not in set(options):
                    raise ValueError(f'{selection} must be None or in {options}')
                else:
                    l.append(selection)
        if 'compare_parameters' in selection:
            l.append('compare_parameter')

        if not loc:
             loc = ''
        else:
            if loc[-1] != '/':
                loc += '/'

        if 'view_parameters' in selection:
            for t in self.data.keys():
                for d in ['mean_pred','mean_err']:
                    # if t == 'no_edges':
                    #     print('passing no_edges in view_parameters')
                    # else:
                    self.view_parameters(
                        t
                        ,loc
                        ,dep=d
                        ,savefig=savefigs
                        ,res=resolution
                        )
        print('done with view_parameters ')
        # print('uncomment view_parameters while testing other functions')

        if 'compare_parameter' in selection:
            for p in self.parameter_names.keys():
                for d in ['mean_pred','mean_err']:
                    if p == 'topo_p':
                        # only these topologies have topo_p, avoid weird error
                        COMPARE_TOPOS = ['erdos', 'random_digraph','watts2', 'watts4','geom']
                    else:
                        COMPARE_TOPOS = None
                    self.compare_parameter(
                        p,
                        loc,
                        dep = d,
                        savefig = savefigs,
                        res = resolution,
                        verbose = verbose,
                        compare_topos = COMPARE_TOPOS, #None means self.data.keys()
                        )
            print('done with compare_parameter ')
            print('axis labels for compare_parameter?')


        # print('done with number of connected components')
        pass

class Evaluate:
    """This class is for briefly evaluating the datasets to identify bias, nullity, and other unevenness in the data """
    def __init__(self,dir=None,file_list=None):
        """
        parameters:
            dir                         (str): str describing the path to directory where filenames are located, if none then
                                                the path is assumed to be the working directory
            file_list                   (list): list containing file names that should be considered, this gives the user the option
                                                of neglecting some filenames


        """
        self.dir = dir
        self.file_list = file_list
        self.parameter_names = {'adj_size':'Network Size',
          'topo_p': 'Mean Degree / Rewiring Prob',
          'gamma': 'Gamma',
          'sigma': 'Sigma',
          'spect_rad':'Spectral Radius',
          'ridge_alpha': 'Ridge Alpha (Regularization)'}

        print('see the inquiry below')
        """

        how many exp_numbers have fewer than 25 nets for each dataset, what percent of experment numbers arent 25
        what are the unique values for gamma, and sigma, and the value counts
        how many experiment numbers does each dataset have
        what is the null percentage for the

        maybe just get all the features im interested in, with just one function, the constructor,
        dont repeat what some of the visualization does
        get nullvalues first

        """

    def export_results(self,results,filename=None):
        """ """
        print('make sure that results is a dataframe ')
        if filename is None:
            month, day = dt.datetime.now().month, dt.datetime.now().day
            hour, minute = dt.datetime.now().hour, dt.datetime.now().minute
            filename = f'datasets_evaluation_{month}_{day}_at_{hour}_{minute}'
        else:
            if filename[-4:] == '.csv':
                filename = filename[-4:]
        results.to_csv(f'{filename}.csv',index=False)
        print(f'produced {filename}.csv')

    def eval_topo(self,df_dict):
        """Evaluate each topology  """

        results = dict()
        counter = 0
        for topo in df_dict.keys():
            for param in self.parameter_names.keys():
                x = df_dict[topo].copy()
                for val in x[param].unique():
                    results[counter] = {'topo':topo
                        ,'param':param
                        ,'val':val
                        ,'count':x.loc[x[param] == val].shape[0]
                    }
                    counter += 1

        topo_summary = dict()
        for i,topo in enumerate(df_dict.keys()):
            topo_summary[i] = {
                'topo':topo
                ,'row_count':df_dict[topo].shape[0]
            }

        month, day = dt.datetime.now().month, dt.datetime.now().day
        hour, minute = dt.datetime.now().hour, dt.datetime.now().minute

        self.export_results(pd.DataFrame(results).T,filename=f'deep_topo_eval_{month}_{day}_at_{hour}_{minute}')
        self.export_results(pd.DataFrame(topo_summary).T,filename=f'topo_summary_{month}_{day}_at_{hour}_{minute}')

    def all(self):
        """ """
        dir = self.dir
        file_list = self.file_list
        get_nullity_for_columns = ['mean_pred', 'mean_err', 'adj_size', 'topo_p', 'gamma', 'sigma',
                   'spect_rad', 'ridge_alpha', 'remove_p', 'pred', 'err', 'max_scc',
                   'max_wcc', 'giant_comp', 'singletons', 'nwcc', 'nscc', 'cluster',
                   'assort', 'diam']

        if file_list is None:
            file_list = os.listdir(dir)

        path = ''
        if dir:
            if dir[-1] != '/':
                path = dir + '/'
            else:
                path = dir


        results = dict()
        for i,f in enumerate(file_list):
            # there could be other files that shouldnt be considered
            if 'compiled' in f and f[-4:] == '.pkl':
                results[i] = dict()
                #read in the as dataframe
                df = pd.DataFrame(pickle.load(open(path + f,'rb')))
                df.columns = [a.lower().replace(' ','_') for a in df.columns]

                #get null value counts then drop

                results[i]['filename'] = f

                for col in get_nullity_for_columns:
                    try:
                        name = f'null_' + col + '_percent'
                        results[i][name] = round(df[col].isna().sum() / df[col].shape[0],3) * 100
                    except:
    #                     print(col,'didnt work -',f)
                        pass
                #drop the null to help `net` out
                df.drop(index=df[df['adj_size'].isnull()].index,inplace=True)
                df.drop(index=df[df['adj_size'] == 0 ].index,inplace=True)
                df.drop(index=df[df['gamma'] == 0 ].index,inplace=True)
                try:
                    #some of the dataframes dont have net
                    results[i]['net'] = df.net.unique()
                except:
                    pass
                row_count = df.shape[0]
                results[i]['net_count_millions'] = row_count / int(1e6)
                results[i]['exp_num_count'] = len(df.exp_num.unique())
                try:
                    results[i]['gamma_value_count_percent'] = round(df.gamma.value_counts().sort_values(inplace=True) / row_count,3) * 100
                    results[i]['sigma_value_count_percent'] = round(df.sigma.value_counts().sort_values(inplace=True)  / row_count,3) * 100
                    results[i]['topo_p_value_count_percent'] = round(df.topo_p.value_counts().sort_values(inplace=True)  / row_count,3) * 100
                    results[i]['remove_p_value_count_percent'] = round(df.remove_p.value_counts().sort_values(inplace=True)  / row_count,3) * 100
                except:
                    # these parameters aren't essential
                    pass
                # to see the next values once finished then use this command: df['topo_p_vals%'].values[0]

        #transpose the dictionary
        self.export_results(pd.DataFrame(results).T)

    def total_net_counts(self):
        """Identify how many experiments (25 nets) there are per remove value """
        pass

    def contain_net_stats(self):
        """ """
        pass

def df_dict(
    drop_partial_experiments=None,
    drop_values=None,
    combine_topos=False,
    scale_down=False,
    file_list=None,
    directory=None
):
    """
    Build a dictionary with topology dataframes which will serve as input for both Visualize & Optimize
    - Merge the data by topology

    parameters:
        directory                         (str): str describing the path to directory where filenames are located, if none then
                                            the path is assumed to be the working directory
        file_list                   (list): list containing file names that should be considered, this gives the user the option
                                            of neglecting some filenames
        drop_partial_experiments    (int): if None, then dont drop the partial experiments, otherwise drop all the nets that
                                            have fewer than this value in num_nets_by_exp
        combine_topos               (bool): if true then watts3 will be considered watts2, and watts5 will become watts4
        scale_down                  (bool): if true then when reading in the data, scale it down for testing purposes
    """
    filenames = {
         'barab1':[]
         ,'barab2':[]
         ,'erdos':[]
         ,'random_digraph':[]
         ,'ident':[]
         ,'watts2':[]
         ,'watts4':[]
         ,'watts3':[] #will become watts2 as well
         ,'watts5':[] #will become watts4 as well
         ,'geom':[]
         ,'no_edges':[]
         ,'loop':[]
         ,'chain':[]
     }

    if file_list is None:
        if directory == '':
            file_list = os.listdir()
        else:
            file_list = os.listdir(directory)

    #cast as a set then as a list, to avoid duplicates
    for i in list(set(file_list)):
        for j in filenames.keys():
            if j in i and i[-4:] == '.pkl':
                filenames[j].append(i)

    change_topos = {'watts3':'watts2','watts5':'watts4'} #used in edit_df as well
    if combine_topos:
        for topo in change_topos.keys():
            for i in filenames[topo]:
                filenames[change_topos[topo]].append(i)


    path = ''
    if directory:
        if directory[-1] != '/':
            path = directory + '/'
        else:
            path = directory

    def edit_df(df,filename,net,drop_partial_experiments=None):
        """
        helper function for df_dict construction

        parameters
            df                          (pandas.DataFrame): dataframe to edit
            filename                    (str): filename essential to track which datasets dont have the net feature
            net                         (str): if net isnt in dataframe already, then this should impute it,
            drop_partial_experiments    (int): if None, then dont drop the partial experiments, otherwise drop all the nets that
                                                have fewer than this value in num_nets_by_exp


        returns:
            df   (pandas.DataFrame): edited dataframe
        """
        df.columns = [a.lower().replace(' ','_') for a in df.columns]
        df.drop(index=df.loc[df['adj_size'].isnull()].index,inplace=True)
        df.drop(index=df.loc[df['adj_size'] == 0 ].index,inplace=True)
        df.drop(index=df.loc[df['gamma'] == 0 ].index,inplace=True)
        s = df['exp_num'].value_counts()
        df.loc[:,'num_nets_by_exp'] = [s[i] for i in df['exp_num']]
        df['num_orbits'] = [len(x) for x in df.err]
        df['max_orbit_pred'] = [max(x) for x in df.pred]
        df['orbit_pred_stdv'] = [np.var(x)**(1/2) for x in df.pred]
        if drop_partial_experiments:
            maximum = df['num_nets_by_exp'].max()
            # if drop_partial_experiments > df['num_nets_by_exp'].max():
            #     print('working with file:',filename)
            #     raise ValueError(f'cant drop {drop_partial_experiments} which is more than the max of {maximum}')
            # else:
                #drop the networks
            # this drop has been added into the optimization class constructor
            # df.drop(index=df[df['num_nets_by_exp'] < drop_partial_experiments ].index,inplace=True)
            if len(df[df['num_nets_by_exp'] < drop_partial_experiments ].index) > 0:
                print(f'{filename} has experiments that dont have {drop_partial_experiments} nets per experiment\n and the max is {maximum}')
        if "net" not in df.columns:
            print(f'`net` column wasnt in:\n{filename}')
            if net is None:
                raise ValueError('Net wasnt specified for this dataset which needs it',filename)
            else:
                df.loc[:,'net'] = net
        if scale_down:
            a = df.shape[0]
            # if a > int(1e3):
            #     # df = df.sample(int(1e3))
            #     df = df.iloc[:int(1e3)]
            # elif a > int(1e2):
            if a > int(1e2):
                # df = df.sample(int(1e3))
                df = df.iloc[:int(1e2)]
            else:
                #if less than 100 rows then it's small enough
                pass

        return df

    df_dict = dict()
    #check to see if there is a key that is unexpected (unexpected topology )
    network_options = {'barab1', 'barab2','barab4',
                    'erdos', 'random_digraph',
                    'watts3', 'watts5',
                    'watts2','watts4',
                    'geom', 'no_edges',
                    'loop', 'chain',
                    'ident'
                  }
    for i in filenames.keys():
        if i not in network_options:
            raise ValueError(f'{i} is not a valid topology in \n{network_options}')

    #create one dataframe per topology
    for i in filenames.keys():
        #save is a boolean to avoid errors in creating dataframes unnecessarily, df_dicts only contains df's not None's
        save = False
        l = filenames[i]
        if len(l) == 0:
            df = None
        elif len(l) == 1:
            save = True
            try:
                a = pd.DataFrame(pickle.load(open(path + l[0],'rb')))
                df = edit_df(a,l[0],i,drop_partial_experiments)
            except:
                print(l[0],'failed probably cuz its an empty file w/ 0 bytes')
        elif len(l) > 1:
            save = True
            #initialize starting with one
            a = pd.DataFrame(pickle.load(open(path + l[0],'rb')))
            #drop_partial_experiments could be None
            df = edit_df(a,l[0],i,drop_partial_experiments)
            for j in l[1:]:
                try:
                    # use pandas instead of merge_compiled for cases when one
                    # when one dataset has net_stats and other datasets do not
                    b = pd.DataFrame(pickle.load(open(path + j,'rb')))
                    b_df = edit_df(b,j,i,drop_partial_experiments)
                    # add the exp_numbers of the dataframes before concatenating to maintain effective differentiation of experiments
                    b_df.loc[:,'exp_num'] += df.exp_num.max() #not off by one
                    df = pd.concat([df,b_df],ignore_index=True,sort=False)
                except:
                    print(j,'failed probably cuz its an empty file w/ 0 bytes')

        if save:
            df_dict[i] = df

    #make last changes to dataframes
    if combine_topos:
        #reassign net column
        for topo in change_topos.values():
            #make sure there is data for it to avoid error
            if topo in df_dict.keys():
                x = df_dict[topo]
                # we have other topologies in x so reassign the net column
                x.loc[:,'net'] = topo
                #df[topo] = x #the change is inplace, these dataframes (dictionary values) are mutable
        #go through the topologies that got merged, & delete them
        for topo in change_topos.keys():
            #make sure there is data for it to avoid error
            if topo in df_dict.keys():
                df_dict.pop(topo)


    return df_dict

def main(selection=None,drop_values=DROP_VALUES):
    """ Use df_dict, to initialize Visualize & Optimize classes, and to run the `all` method for each class
    The selection parameter is useful when wanting to run just one of 'visualize' or 'optimize'

    Parameters:
        selection   (str): If None, then both Visualize & Optimize
        drop_values (dict of lists): parameter for df_dicts, see docstring there
    """
    options = ['visualize','optimize','evaluate']
    l = []
    if selection is None or selection == 'None':
        l = options
    else:
        if isinstance(selection,list):
            for i in selection:
                if i not in set(options):
                    raise ValueError(f'{i} must be None or in {options}')
                else:
                    l.append(i)
        else:
            if selection not in set(options):
                raise ValueError(f'{selection} must be None or in {options}')
            else:
                l.append(selection)

    #make loc directory
    subprocess.run(['mkdir',LOC])


    # DIR FILE_LIST parameters defined at top of script, for easy modification in VIM
    start = time.time()
    d = df_dict(
        drop_partial_experiments=DROP_PARTIAL_EXPERIMENTS
        ,drop_values=DROP_VALUES
        ,combine_topos=combine_topos
        ,scale_down=SCALE_DOWN
        ,file_list=FILE_LIST
        ,directory=DIR
    )
    print(f'DF_DICT construction time (minutes):',round((time.time() - start )/ 60,1))

    if 'visualize' in l:
        start = time.time()
        run_all = ['view_parameters','compare_parameter','compare_parameters']
        V = Visualize(d,drop_values=drop_values)
        V.all(selection=run_all
            ,savefigs=SAVEFIGS
            ,resolution=RESOLUTION
            ,loc=LOC,verbose=True)
        print(f'Visualization time (minutes):',round((time.time() - start )/ 60,1))
    if 'evaluate' in l:
        start = time.time()
        e = Evaluate(dir=DIR,file_list=FILE_LIST)
        e.eval_topo(d)
        e.all()
        print(f'Evaluation time (minutes):',round((time.time() - start )/ 60,1))
    if 'optimize' in l:
        start = time.time()
        O = Optimize(d)
        results = O.win(NUM_WINNERS,LOC)
        print(f'Optimization time (minutes):',round((time.time() - start )/ 60,1))
        return results

def test_visuals():
    """ """
    d = df_dict(DIR,FILE_LIST)
    V = Visualize(d)
    V.compare_parameter(
        'spect_rad',
        loc=None,
        dep = 'mean_pred',
        savefig = False,
        res = int(1e2),
        verbose = False,
        compare_topos = ['barab1','watts3','loop'],
    )
    V.view_parameters('watts4',savefig=False) #changed parameters for view_parameters

print('test the drop values both when the lists are empty, incorrect values, and when it should be correct ')

main(SELECTION)
# test_visuals()
#find the file

print('if DIR = None, then building the dataframes with a path can cause problems')
