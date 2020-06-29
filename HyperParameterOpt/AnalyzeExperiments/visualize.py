def compare_parameters(
    t = 'erdos',
    dep = 'mean_pred',
    res = int(1e3),
    verbose = False
    ):
    """
    Create visualizations comparing parameter values influence on dependent variable 
    as well as the distribution of number of nets for each remove_p per value for each 
    parameter (as a logarithmic line plot)
    
    Parameters: 
        t       (str): topology  
        dep     (str): dependent variable
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

    month, day = dt.datetime.now().month, dt.datetime.now().day
    hour, minute = dt.datetime.now().hour, dt.datetime.now().minute

    if dep == 'mean_pred':
        fig.savefig(f'Visuals/Optimize_{t}_{month}_{day}_at_{hour}_{minute}.png',bbox_inches='tight',bbox_extra_artists=[my_suptitle,leg0,leg1])
    else:
        fig.savefig(f'Visuals/Optimize_{t}_by_fit_{month}_{day}_at_{hour}_{minute}.png',bbox_inches='tight',bbox_extra_artists=[my_suptitle,,leg0,leg1])

        
