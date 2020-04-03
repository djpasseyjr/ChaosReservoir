DIR = "Watts"
FILE_START = "watts_"
TMPL = "template.py"
BASH_TMPL = "bash_template.sh"

def mass_prod_exp(exp_name, net, remove_ps, nfile_per_exp, nexisting_files=0):
    """ Write experiment files with the given parameter settings
    """
    exp_fnames = []
    for j,p in enumerate(remove_ps):
        for i in range(nexisting_files+1, nexisting_files + nfile_per_exp+1):
            fname      = FILE_START + exp_name + '_'+str(p)+'_'+str(i)
            exp_fname  = fname + '.py'
            save_fname = fname + '.pkl'
            make_exp_file(DIR + '/' + exp_fname, save_fname, net, str(p))
            make_bash_file(exp_fname)
            
def make_exp_file(exp_fname, save_fname, net, remove_p):
    """ Make a single experiment file
    """
    tmpl_stream = open(TMPL,'r')
    tmpl_str = tmpl_stream.read()
    tmpl_str = tmpl_str.replace("#FNAME#",save_fname)
    tmpl_str = tmpl_str.replace("#NET#",net)
    tmpl_str = tmpl_str.replace("#REMOVE_P#",remove_p)
    # Save to new file
    new_f = open(exp_fname,'w')
    new_f.write(tmpl_str)
    new_f.close()
    
def make_bash_file(fname):
    """ Make a bash file to run 'fname' on the super computer
    """
    # Open bash template
    tmpl_stream = open(BASH_TMPL,'r')
    tmpl_str = tmpl_stream.read()
    tmpl_stream.close()
    
    # Get current experiment number
    idx = tmpl_str.find('\n')
    exp_num = tmpl_str[:idx]
    contents = tmpl_str[idx+1:]
    bash_fname = DIR + '/'+'run_exp' + exp_num +'.sh'
    
    # Increase experiment number by one
    tmpl_stream = open(BASH_TMPL,'w')
    new_bash_tmpl = str(int(exp_num)+1) + "\n" + contents
    tmpl_stream.write(new_bash_tmpl)
    tmpl_stream.close()
    
    # Adjust the new bash file to run fname
    contents = contents.replace("#FNAME#",fname)
    # Save to new file
    new_f = open(bash_fname,'w')
    chars = new_f.write(contents)
    new_f.close()
    