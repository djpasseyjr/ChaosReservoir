import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
import pickle
import datetime as dt                   # to add the date and time into the figure title
import time

# DIR = '/Users/joeywilkes/ReservoirComputing/research_data'
FILE_LIST = [
    'compiled_output_f2_38_barab2.pkl'
    ,'compiled_output_jw39_barab1.pkl'
    'compiled_output_jw40_watts3.pkl'
    ,'compiled_output_jw45_watts3.pkl'
    ,'compiled_output_jw53_ident.pkl'
    ,'compiled_output_jw54_loop.pkl'
    ,'compiled_output_jw55_no_edges.pkl'
    ,'compiled_output_jj7_erdos.pkl'
    ,'compiled_output_jw56_chain.pkl'
    ,'compiled_output_jj6_random_digraph.pkl'
    ,'compiled_output_jw39_barab1.pkl'
    ,'compiled_output_jw43_barab1.pkl'
]

SAVEFIGS = True
RESOLUTION = int(1e3)
DIR = None
# FILE_LIST = None
NUM_WINNERS = 5 #find top NUM_WINNERS in grid search optimization

#selection for either 'visualize' or 'optimize' or None (means both)
SELECTION = 'optimize'
# LOCATION FOR OUTPUT FILES
LOC = 'BEST_DATA_AS_OF_7_15'

class Visualize:
    """Visualization tool"""

    def __init__(self,df_dict):
        """
        Initialize data members for visualizations

        Parameters:
            df_dict     (dict): keys are topology name strings, and values are the dataframe for
                                that topology, aka output of df_dict function
        """
        self.data = df_dict
        #used in compare_parameters (at least)
        self.parameter_names = {'adj_size':'Network Size',
          'topo_p': 'Mean Degree / Rewiring Prob',
          'gamma': 'Gamma',
          'sigma': 'Sigma',
          'spect_rad':'Spectral Radius',
          'ridge_alpha': 'Ridge Alpha (Regularization)'}
        self.topo_names = {
             'barab1':'Barabasi One'
             ,'barab2':'Barabasi Two'
             ,'erdos':'Erdos Reyni'
             ,'random_digraph':'Random Directed Graph'
             ,'ident':'Identity'
             ,'watts2':'Watts Strogatz 2'
             ,'watts4':'Watts Strogatz 4'
             ,'watts3':'Watts Strogatz 3'
             ,'watts5':'Watts Strogatz 4'
             ,'geom':'Random Geometric'
             ,'no_edges':'No Edges'
             ,'loop':'Loop Network'
             ,'chain':'Chain Network'
         }
        self.var_names = {
        'mean_pred':'Avg. Accuracy Dur.'
        ,'mean_err':'Avg. Fit Error'
        ,'remove_p':'Edge Removal %'
        ,'ncd':'# Nets (log)' #net count distribution
        }
        self.axis_names
        self.legend_size = 10
        # self.figure_width_per_column = 6.5
        self.figure_width_per_column = 9 # when legend is bigger (legend_size = 10)
        self.figure_height_per_row = 3 #or low as 2.5

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
        x,
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
            x       (df):  dataframe containing the data
            t       (str): topology
            loc     (str): location / directory for output files
            dep     (str): dependent variable
            savefig (str): location to save figure, if None, then don't save the figure
            res     (int): resolution of images
            verbose (str): debugging purposes
        """
        hyp_p = self.parameter_names.keys()
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

        if not loc:
             loc = ''
        else:
            if loc[-1] != '/':
                loc += '/'

        if t not in network_options:
            raise ValueError(f'{t} not in {network_options}')

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
            for j,p in enumerate(x[v].unique()):
                if verbose:
                    print(i,v,j,p,'\n')
                S = e[e[v] == p].groupby(e.remove_p).aggregate(np.mean)[dep].copy()
                ax[i][0].plot(S.index,S.values,label=p) #if one topology
                ax[i][0].scatter(S.index,S.values)

                A = pd.DataFrame(x.loc[(x.net == t) & (x[v] == p)]['remove_p'].value_counts())
                A.reset_index(inplace=True)
                A.sort_values(by='index',inplace=True)
                ax[i][1].semilogy(A['index'],A['remove_p'],label=p)

            leg0 = ax[i][0].legend(prop={'size': self.legend_size},bbox_to_anchor=(-0.2, 0.5))
            ax[i][0].set_title(f'{self.parameter_names[v]} Value Comparison')
            ax[i][0].set_xlabel(self.var_names['remove_p'])
            ax[i][0].set_ylabel(self.var_names[dep])

            leg1 = ax[i][1].legend(prop={'size': self.legend_size},bbox_to_anchor=(1.2, 0.5))
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
            for j,p in enumerate(e[v].unique()):
                S = e[e[v] == p].groupby(e.remove_p).aggregate(np.mean)[dep].copy()
                ax[i].plot(S.index,S.values,label=p) #if one topology
                ax[i].scatter(S.index,S.values)
            leg0 = ax[i].legend(loc='lower left',prop={'size': self.legend_size},bbox_to_anchor=(1.1, 0.5))
            ax[i].set_title(self.topo_names[t])
            ax[i].set_xlabel(self.var_names['remove_p'])
            ax[i].set_ylabel(self.var_names[dep])
        print('we need axis labels')
        # title_ = v.upper().replace('_',' ')
        # fig.suptitle(f'{title_} Comparison For All Topologies', fontsize=16,y=1.03)
        my_suptitle = fig.suptitle(f'{self.parameter_names[v]} Comparison For All Topologies', fontsize=16,y=1.03)
        plt.tight_layout()
        # plt.show()

        if savefig:
            month, day = dt.datetime.now().month, dt.datetime.now().day
            hour, minute = dt.datetime.now().hour, dt.datetime.now().minute
            if dep == 'mean_pred':
                fig.savefig(loc + f'Compare_{parameter}_{month}_{day}_at_{hour}_{minute}.png',bbox_inches='tight',bbox_extra_artists=[my_suptitle,leg0])
            else:
                fig.savefig(loc + f'Compare_{parameter}_by_fit_{month}_{day}_at_{hour}_{minute}.png',bbox_inches='tight',bbox_extra_artists=[my_suptitle,leg0])

    def contrast_topos(self):
        """Plot all the topologies against each other """
        print('how to limit to either a subset of parameters, or best')
        raise NotImplementedError('contrast_topos not done ')

    def network_statistics(self):
        """ """
        raise NotImplementedError('network_statistics not done ')

    def all(self
        ,savefigs=True
        ,resolution=int(1e2)
        ,loc=None
        ,verbose=False):
        """ The method will will create all the visuals

        Parameters
            savefigs    (bool): save the figures
            resolution  (int): resolution for the figures
            loc         (str): location / directory for output files
            verbose     (str): verbose output

        """
        if not loc:
             loc = ''
        else:
            if loc[-1] != '/':
                loc += '/'

        for t in self.data.keys():
            for d in ['mean_pred','mean_err']:
                self.view_parameters(
                    self.data[t]
                    ,t
                    ,loc
                    ,dep=d
                    ,savefig=savefigs
                    ,res=resolution
                    )
        print('done with view_parameters ')
        # print('uncomment view_parameters while testing other functions')

        for p in self.parameter_names.keys():
            for d in ['mean_pred','mean_err']:
                self.compare_parameter(
                    p,
                    loc,
                    dep = d,
                    savefig = savefigs,
                    res = resolution,
                    verbose = verbose,
                    compare_topos = None,
                    )
        print('done with compare_parameter ')
        print('axis labels for compare_parameter')

class Optimize:
    """ """
    def __init__(self,df_dict):
        """
        Parameters:
            df_dict     (dict): keys are topology name strings, and values are the dataframe for
                                that topology, aka output of df_dict function
        """
        self.data = df_dict
        self.topos = dict()
        self.compare = dict()
        self.best = dict()

    def win(self,num_winners=5,loc=None):
        """
        Get top `num_winners` from each topology for both model types (thinned or not thinned)
        as well as winners out of any topology (comparing) by model type (thinned or not thinned)
        as well as winners out of all model types out of all topologies (the best model from this data)

        Assume "dense" means not thinned in this case, remove_p = 0

        Parameters:
            num_winners     (int): number of winners from each topology to consider
            loc             (str): location/directory for output file

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
            df = self.topos[list(self.data.keys())[0]][m]
            df['net'] = list(self.data.keys())[0]
            for i in list(self.data.keys())[1:]:
                temp = self.topos[i][m]
                temp['net'] = i
                df = df.append(temp,ignore_index=False)
            df.sort_values(by=['mean_pred','mean_err'],ascending=[False,True],inplace=True)
            self.compare[m] = df
            best = best.append(df,ignore_index=False)

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
        name = loc + file
        df.to_pickle(name)

        return results

def merge_compiled(compiled1, compiled2):
   """ Merge two compiled dictionaries """
   if isinstance(compiled1, str) and isinstance(compiled2, str):
       compiled1 = pickle.load(open(compiled1, 'rb'))
       compiled2 = pickle.load(open(compiled2, 'rb'))
   # Shift experiment number for compiled2
   total_exp = np.max(compiled1["exp_num"])
   exp_nums = np.array(compiled2["exp_num"])
   exp_nums[exp_nums >= 0] += total_exp
   compiled2["exp_num"] = list(exp_nums)
   # Merge
   for k in compiled1.keys():
       compiled1[k] += compiled2[k]
   return compiled1

def df_dict(dir=None,file_list=None):
    """
    Build a dictionary with topology dataframes which will serve as input for both Visualize & Optimize
    - Merge the data by topology

    parameters:
        dir        (str): str describing the path to directory where filenames are located, if none then
                            the pa«th is assumed to be the working directory
        file_list  (list): list containing file names that should be considered, this gives the user the option
                            of neglecting some filenames
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
        file_list = os.listdir(dir)

    #cast as a set then as a list, to avoid duplicates
    for i in list(set(file_list)):
        for j in filenames.keys():
            if j in i:
                filenames[j].append(i)

    path = ''
    if dir:
        if dir[-1] != '/':
            path = dir + '/'
        else:
            path = dir

    print('try where dir is none, and otherwise')

    df_dict = dict()
    #check to see if there is a key that is unexpected (unexpected topology )
    network_options = {'barab1', 'barab2',
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
        edit = False #boolean to avoid errors in creating dataframe
        l = filenames[i]
        if len(l) == 0:
            pass
        elif len(l) == 1:
            edit = True
            a = pickle.load(open(path + l[0],'rb'))
        elif len(l) > 1:
            edit = True
            #initialize starting with one
            a = pickle.load(open(path + l[0],'rb'))
            for j in l[1:]:
                a = merge_compiled(a,pickle.load(open(path + j,'rb')))

        if edit:
            df = pd.DataFrame(a)
            df.columns = [a.lower().replace(' ','_') for a in df.columns]
            df.drop(index=df[df['adj_size'].isnull()].index,inplace=True)
            if "net" not in df.columns:
                df['net'] = i
                print(f'`net` column wasnt in:\n{filenames[i]}')
            df_dict[i] = df

    return df_dict

def main(selection=None):
    """ Use df_dict, to initialize Visualize & Optimize classes, and to run the `all` method for each class
    The selection parameter is useful when wanting to run just one of 'visualize' or 'optimize'

    Parameters:
        selection (str): If None, then both Visualize & Optimize
    """
    options = ['visualize','optimize']
    if selection is None:
        l = options
    else:
        if selection not in options:
            raise ValueError(f'{selection} must be None or in {options}')
        else:
            l = [selection]


    # DIR FILE_LIST parameters defined at top of script, for easy modification in VIM
    start = time.time()
    d = df_dict(DIR,FILE_LIST)
    print(f'DF_DICT construction time (minutes):',round((time.time() - start )/ 60,1))

    if 'visualize' in l:
        start = time.time()
        V = Visualize(d)
        V.all(SAVEFIGS,RESOLUTION,LOC)
        print(f'Visualization time (minutes):',round((time.time() - start )/ 60,1))
    if 'optimize' in l:
        start = time.time()
        O = Optimize(d)
        results = O.win(NUM_WINNERS,LOC)
        print(f'Optimization time (minutes):',round((time.time() - start )/ 60,1))
        return results

print('how to exclude certain parameter values ? ')

if __name__ == "__main__":
    main(SELECTION)
