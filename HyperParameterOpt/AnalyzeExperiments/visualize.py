import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
import pickle
import datetime as dt                   # to add the date and time into the figure title

print('should I include boolean for plt.show - like if we just wanted to save the figures & not see them (like while running in super-computer) ?')

class Visualize:
    """Visualization tool"""

    def __init__(self,dir=None,file_list=None):
        """
        Merge the data by topology

        parameters:
            dir        (str): str describing the path to directory where files are located, if none then
                                the path is assumed to be the working directory
            file_list  (list): list containing file names that should be considered, this gives the user the option
                                of neglecting some files

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
             ,'ident':[]
         }

        if file_list is None:
            file_list = os.listdir(dir)

        #cast as a set then as a list, to avoid duplicates
        for i in list(set(file_list)):
            for j in filenames.keys():
                if j in i:
                    filenames[j].append(i)

        self.filenames = filenames
        if dir:
            if dir[-1] != '/':
                path = dir + '/'
        else:
            #to simply upcoming code
            path = ''

        print('try where dir is none, and otherwise')
        print('ask if sharedy should be a parameter for everything, or just some things')

        self.data = dict()
        #check to see if there is a key that is unexpected (unexpected topology )
        network_options = {'barab1', 'barab2',
                        'erdos', 'random_digraph',
                        'watts3', 'watts5',
                        'watts2','watts4',
                        'geom', 'no_edges',
                        'loop', 'chain',
                        'ident'
                      }
        for i in files.keys():
            if i not in network_options:
                raise ValueError(f'{i} is not a valid topology in \n{network_options}')

        #create one dataframe per topology
        for i in files.keys():
            l = files[i]
            if len(l) == 0:
                pass
            elif len(l) == 1:
                self.data[i] = pd.DataFrame(pickle.load(open(path + l[0],'rb')))
            elif len(l) > 1:
                #initialize starting with one
                a = pickle.load(open(path + l[0],'rb'))
                for j in l[1:]:
                    a = self.merge_compiled(a,pickle.load(open(path + j,'rb')))
                self.data[i] = pd.DataFrame(a)

        #used in compare_parameters (at least)
        self.parameter_names = {'adj_size':'Network Size',
          'topo_p': 'Mean Degree / Rewiring Prob',
          'gamma': 'Gamma',
          'sigma': 'Sigma',
          'spect_rad':'Spectral Radius',
          'ridge_alpha': 'Ridge Alpha (Regularization)'}

    def merge_compiled(self,compiled1, compiled2):
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
            dep     (str): dependent variable
            savefig (str): location to save figure, if None, then don't save the figure
            res     (int): resolution of images
            verbose (str): debugging purposes
        """
        hyp_p = ['adj_size', 'topo_p', 'gamma', 'sigma',
           'spect_rad', 'ridge_alpha']
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

        if t not in network_options:
            raise ValueError(f'{t} not in {network_options}')

        if dep not in ['mean_pred','mean_err']:
            print(f'{dep} is not acceptable as dependent variable ')

        #HS is for hspace
        # HS = 2
        fig_height = 3 * len(hyp_p)
        fig_width = 13
        fig, ax = plt.subplots(len(hyp_p),2,sharey=False,dpi=resolution,figsize=(fig_width,fig_height))
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
                ax[i][1].semilogy(A['index'],A['remove_p'],label=p)

            leg0 = ax[i][0].legend(prop={'size': 5},bbox_to_anchor=(-0.2, 0.5))
            ax[i][0].set_title(f'{v} Value Comparison')
            ax[i][0].set_xlabel('remove_p')

            leg1 = ax[i][1].legend(prop={'size': 5},bbox_to_anchor=(1.2, 0.5))
            ax[i][1].set_xlabel('remove_p')
            ax[i][1].set_ylabel('# nets (log)')
            ax[i][1].set_title(f'{v} value counts per value')

            if dep == 'mean_pred':
                ax[i][0].set_ylabel('Mean pred')
            else:
                ax[i][0].set_ylabel('Mean Err')

        my_suptitle = fig.suptitle(f'{t} Hyper-Parameter Comparison', fontsize=16,y=1.03)
        plt.tight_layout()
        plt.show()

        if savefig:
            month, day = dt.datetime.now().month, dt.datetime.now().day
            hour, minute = dt.datetime.now().hour, dt.datetime.now().minute
            if dep == 'mean_pred':
                fig.savefig(loc + f'Optimize_{t}_{month}_{day}_at_{hour}_{minute}.png',bbox_inches='tight',bbox_extra_artists=[my_suptitle,leg0,leg1])
            else:
                fig.savefig(loc + f'Optimize_{t}_by_fit_{month}_{day}_at_{hour}_{minute}.png',bbox_inches='tight',bbox_extra_artists=[my_suptitle,leg0,leg1])

    def compare_parameter(self,
        parameter,
        dep = 'mean_pred',
        savefig = False,
        res = int(1e2),
        verbose = False,
        compare_all = None,
        ):
        """For a given Parameter, compare the topologies
        display all the topologies

        Parameters:
            parameter   (str): string describing the parameter to compare
            dep         (str): dependent variable
            savefig     (bool): whether or not to export the figure
            res         (int): resolution of images
            verbose     (str): debugging purposes
            compare_all (list): indicates which topologies to include in the comparison, if None then compare all available in self.data

        """
        input = dict()
        input['t'] = t
        input['dep'] = dep
        input['savefig'] = savefig
        input['res'] = res
        tol_results = self.tolerance(input)
        dep = tol_results['dep']
        resolution = tol_results['res']
        parameter = tol_results['parameter']

        if compare_all:
            num_unique_nets = len(compare_all)
        else:
            num_unique_nets = len(self.data.keys())
            compare_all = self.data.keys()


        fig_height = num_unique_nets * 2.5
        fig_width = 6.4

        fig, ax = plt.subplots(num_unique_nets,1,sharey=False,dpi=resolution,figsize=(fig_width,fig_height))

        #each subplot is a topology
        #then all values for the hyper-parameter are compared

        v = parameter

        for i,t in enumerate(compare_all):
            e = self.data[t].copy()
            for j,p in enumerate(e[v].unique()):
                if verbose:
                    print(i,v,j,p,'\n')
                S = e[e[v] == p].groupby(e.remove_p).aggregate(np.mean)[dep].copy()
                ax[i].plot(S.index,S.values,label=p) #if one topology
                ax[i].scatter(S.index,S.values)
            leg0 = ax[i].legend(loc='lower left',prop={'size': 5},bbox_to_anchor=(1.1, 0.5))
            ax[i].set_title(t)
        # title_ = v.upper().replace('_',' ')
        # fig.suptitle(f'{title_} Comparison For All Topologies', fontsize=16,y=1.03)
        my_suptitle = fig.suptitle(f'{self.parameter_names[v]} Comparison For All Topologies', fontsize=16,y=1.03)
        plt.tight_layout()

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
        ,savefigs = False
        ,resolution=int(1e2)):
        """ The method will will create all the visuals """

        for i in self.data.keys():
            for d in ['mean_pred','mean_err']:
                self.view_parameters(
                    self.data[i]
                    ,i
                    ,dep = d
                    ,savefig=savefigs
                    ,res=resolution
                    )
        print('done with view_parameters ')

        for v in self.parameter_names:
            for d in ['mean_pred','mean_err']:
                def compare_parameter(self,
                    v,
                    dep = d,
                    savefig = savefigs,
                    res = int(1e2),
                    verbose = resolution,
                    compare_all = None,
                    ):

        print('done with view_parameters ')

        print('done with view_parameters')


# visualization tools & functions

#Visuals I've done so far
# comapre topologies on one plot (like with additional topologies )
