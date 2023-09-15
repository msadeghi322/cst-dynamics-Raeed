#%%
import src

import pyaldata
import numpy as np
import pandas as pd
import yaml

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupShuffleSplit
from src.models import SSA

import seaborn as sns
import matplotlib.pyplot as plt
import k3d
import scipy.io as sio

with open("../params.yaml", "r") as params_file:
    lfads_params = yaml.safe_load(params_file)["lfads_prep"]

load_params = {
    'file_prefix': 'Prez_20220721',
    'verbose': False,
    'keep_unsorted': False,
    'lfads_params': lfads_params,
    'epoch_fun': src.util.generate_realtime_epoch_fun(
        start_point_name='idx_ctHoldTime',
        end_point_name='idx_endTime',
    ),
    'bin_size': 0.01,
}
td = (
    src.data.load_clean_data(**load_params)
    .query('task=="RTT" | task=="CST"')
    .assign(**{'trialtime': lambda x: x['Time from go cue (s)']})
    .pipe(pyaldata.soft_normalize_signal,signals=['lfads_rates','MC_rates'])
    .pipe(src.data.remove_baseline_rates,signals=['MC_rates','lfads_rates'])
    .pipe(pyaldata.dim_reduce,TruncatedSVD(n_components=15),'MC_rates','MC_pca')
    .pipe(pyaldata.dim_reduce,TruncatedSVD(n_components=15),'lfads_rates','lfads_pca')
    # .pipe(pyaldata.dim_reduce,SSA(R=15,n_epochs=3000,lr=0.01),'MC_rates','MC_ssa')
    # .pipe(pyaldata.dim_reduce,SSA(R=15,n_epochs=3000,lr=0.01),'lfads_rates','lfads_ssa')
)


# %%
trialnum = 228
trial = td.loc[td['trial_id']==trialnum].squeeze()
fig,ax = plt.subplots(4,1,figsize=(10,10))
src.plot.plot_hand_trace(trial,ax=ax[0])
src.plot.plot_hand_velocity(trial,ax=ax[1])
ax[2].plot(trial['lfads_pca'][:,0:15])
ax[3].plot(trial['lfads_inputs'])

# %%
fig,ax = plt.subplots(1,1)
task_colors = {'CST': 'C0','RTT': 'C1'}
signal = 'lfads_pca'
trials_to_plot = [227,228]
for trial_id in trials_to_plot:
    trial = td.loc[td['trial_id']==trial_id].squeeze()
    ax.plot(
        trial[signal][:,0],
        trial[signal][:,1],
        color=task_colors[trial['task']],
    )

sns.despine(ax=ax,left=True,bottom=True)

#%% Context space

signal = 'lfads_rates'
tonic_context_model = LinearDiscriminantAnalysis()
td_models = src.data.rebin_data(td,new_bin_size=0.100)
tonic_context_model.fit(
    np.row_stack(td_models.apply(lambda x: x[signal][x['idx_goCueTime']+15,:],axis=1)),
    td_models['task'],
)

transient_context_model = LinearDiscriminantAnalysis()
transient_context_model.fit(
    np.row_stack(td_models.apply(lambda x: x[signal][x['idx_pretaskHoldTime']+3,:],axis=1)),
    td_models['task'],
)

def norm_vec(vec):
    return vec/np.linalg.norm(vec)

td['Motor Cortex Tonic Context Dim'] = [(sig @ norm_vec(tonic_context_model.coef_).squeeze()[:,None]).squeeze() for sig in td[signal]]
td['Motor Cortex Transient Context Dim'] = [(sig @ norm_vec(transient_context_model.coef_).squeeze()[:,None]).squeeze() for sig in td[signal]]

td_explode = (
    td
    .assign(
        **{'Hand velocity (cm/s)': lambda x: x.apply(lambda y: y['hand_vel'][:,0],axis=1)}
    )
    .filter(items=[
        'trial_id',
        'Time from go cue (s)',
        'Time from task cue (s)',
        'task',
        'Motor Cortex Transient Context Dim',
        'Motor Cortex Tonic Context Dim',
        'Hand velocity (cm/s)'
    ])
    .explode([
        'Time from go cue (s)',
        'Time from task cue (s)',
        'Motor Cortex Transient Context Dim',
        'Motor Cortex Tonic Context Dim',
        'Hand velocity (cm/s)',
    ])
    .astype({
        'Time from go cue (s)': float,
        'Time from task cue (s)': float,
        'Motor Cortex Transient Context Dim': float,
        'Motor Cortex Tonic Context Dim': float,
        'Hand velocity (cm/s)': float,
    })
    # .loc[lambda df: df['Time from go cue (s)']>0]
    # .loc[lambda df: (df['Time from go cue (s)']<0) & (df['Time from go cue (s)']>-0.5)]
)
avg_trial = td_explode.groupby(['Time from go cue (s)','task']).mean().loc[-1:5].reset_index()
task_colors={'RTT': 'C1','CST': 'C0'}
fig,axs = plt.subplots(2,1,sharex=True,figsize=(6,6))
epoch = 'go'
for _,trial in td.groupby('task').sample(n=10).iterrows():
    # put an average trace over this thing
    axs[0].plot(
        trial[f'Time from {epoch} cue (s)'],
        trial['Motor Cortex Transient Context Dim'],
        color=task_colors[trial['task']],
        alpha=0.3,
        lw=2,
    )
    axs[1].plot(
        trial[f'Time from {epoch} cue (s)'],
        trial['Motor Cortex Tonic Context Dim'],
        color=task_colors[trial['task']],
        alpha=0.3,
        lw=2,
    )
    # axs.set_xlim([-1,5])
    # axs.set_ylim([-0.3,0.3])
    # axs.set_ylabel(f'Comp {compnum+1}')
for task,trial in avg_trial.groupby('task'):
    axs[0].plot(
        trial[f'Time from {epoch} cue (s)'],
        trial['Motor Cortex Transient Context Dim'],
        color=task_colors[task],
        lw=4,
    )
    axs[1].plot(
        trial[f'Time from {epoch} cue (s)'],
        trial['Motor Cortex Tonic Context Dim'],
        color=task_colors[task],
        lw=4,
    )
axs[0].set_ylabel('Motor Cortex\nContext Dim')
axs[1].set_ylabel('Motor Cortex\nContext Dim')
#axs[2].set_ylabel('Behavioral\nContext Dim')
axs[-1].set_xlabel(f'Time from {epoch} cue (s)')
axs[0].set_xlim([-1,5])
sns.despine(fig=fig,trim=True)

# %% plot SSA

def plot_trial(trial_to_plot,ax_list,signal_to_plot):
    num_dims = ax_list.shape[0]-1

    beh_ax = ax_list[-1]
    src.plot.plot_hand_trace(trial_to_plot,ax=beh_ax)
    beh_ax.set_ylabel('Pos')

    for i in range(num_dims):
        ax = ax_list[i]
        # Plot SSA results
        ax.plot(trial_to_plot['trialtime'][[0,-1]],[0,0],color='k')
        ax.plot(trial_to_plot['trialtime'],trial_to_plot[signal_to_plot][:,i])
        ax.set_yticks([])
        ax.plot([0,0],ax.get_ylim(),color='k',linestyle='--')
        sns.despine(ax=ax,trim=True)

trials_to_plot = [227,228]
fig,axs = plt.subplots(16,len(trials_to_plot),sharex=True,figsize=(10,10))

for colnum,trial_id in enumerate(trials_to_plot):
    trial = td.loc[td['trial_id']==trial_id].squeeze()
    plot_trial(trial,axs[:,colnum],signal_to_plot='lfads_ssa')


# %% decoding

signal = 'lfads_pca'
def get_test_labels(df):
    gss = GroupShuffleSplit(n_splits=1,test_size=0.25)
    _,test = next(gss.split(
        df[signal],
        df['True velocity'],
        groups=df['trial_id'],
    ))
    return np.isin(np.arange(df.shape[0]),test)

def fit_models(df):
    # individual models
    models = {}
    for task in df['task'].unique():
        models[task] = LinearRegression()
        train_df = df.loc[(~df['Test set']) & (df['task']==task)]
        models[task].fit(
            np.row_stack(train_df[signal]),
            train_df['True velocity'],
        )

    # joint models
    models['Joint'] = LinearRegression()
    train_df = df.loc[~df['Test set']]
    models['Joint'].fit(
        np.row_stack(train_df[signal]),
        train_df['True velocity'],
    )

    return models

def model_predict(df,models):
    ret_df = df.copy()
    for model_name,model in models.items():
        ret_df = ret_df.assign(**{
            f'{model_name} predicted': model.predict(np.row_stack(ret_df[signal]))
        })
    return ret_df

def score_models(df,models):
    scores = pd.Series(index=pd.MultiIndex.from_product(
        [df['task'].unique(),models.keys()],
        names=['Test data','Train data']
    ))
    for task in df['task'].unique():
        for model_name, model in models.items():
            test_df = df.loc[df['Test set'] & (df['task']==task)]
            scores[(task,model_name)] = model.score(np.row_stack(test_df[signal]),test_df['True velocity'])
    
    return scores

td_train_test = (
    td
    .assign(
        **{'True velocity': lambda df: df.apply(lambda s: s['hand_vel'][:,0],axis=1)}
    )
    .filter(items=[
        'trial_id',
        'Time from go cue (s)',
        'task',
        'True velocity',
        signal,
    ])
    .explode([
        'Time from go cue (s)',
        'True velocity',
        signal,
    ])
    .astype({
        'Time from go cue (s)': float,
        'True velocity': float,
    })
    .assign(**{'Test set': lambda df: get_test_labels(df)})
)

models = fit_models(td_train_test)
scores = score_models(td_train_test,models)
td_pred = (
    td_train_test
    .pipe(model_predict,models)
    .melt(
        id_vars=['trial_id','Time from go cue (s)','task'],
        value_vars=['True velocity','CST predicted','RTT predicted','Joint predicted'],
        var_name='Model',
        value_name='Hand velocity (cm/s)',
    )
)

# trials_to_plot=[71,52]
trials_to_plot=td_pred.groupby('task').sample(n=1)['trial_id']
g=sns.relplot(
    data=td_pred.loc[np.isin(td_pred['trial_id'],trials_to_plot)],
    x='Time from go cue (s)',
    y='Hand velocity (cm/s)',
    hue='Model',
    hue_order=['True velocity','CST predicted','RTT predicted','Joint predicted'],
    palette=['k','C0','C1','0.5'],
    kind='line',
    row='trial_id',
    row_order=trials_to_plot,
    height=4,
    aspect=2,
)
g.axes[0,0].set_yticks([-200,0,200])
g.axes[0,0].set_xticks([0,2,4,6])
sns.despine(fig=g.fig,trim=True)

# fig_name = src.util.format_outfile_name(td,postfix='cst71_rtt52_vel_pred')
# g.fig.savefig(os.path.join('../results/2022_sfn_poster/',fig_name+'.pdf'))

fig,ax = plt.subplots(1,1)
sns.heatmap(
    ax=ax,
    data=scores.unstack(),
    vmin=0,
    vmax=1,
    annot=True,
    annot_kws={'fontsize': 21},
    cmap='gray',
)

# %% Export data for subspace splitting in MATLAB
'''
This code will go through the following steps:
- Restrict to only data from -300ms to 5000ms from the go cue
- Run a separated-rejoined PCA on CST and RTT data:
    - Run PCA separately on RTT and CST data in this epoch
    - Concatenate the PC weights from the two tasks
    - Run SVD on the concatenated PC weights to get new component weights
    - Project the data onto the new component weights
- Calculate the covariance matrices for each task
'''

signal = 'lfads_rates'
num_dims = 15
td_trim = (
    td
    .assign(
        **{'Hand velocity (cm/s)': lambda x: x.apply(lambda y: y['hand_vel'][:,0],axis=1)}
    )
    .filter(items=[
        'trial_id',
        'task',
        'Time from go cue (s)',
        'Hand velocity (cm/s)',
        signal,
    ])
    .explode([
        'Time from go cue (s)',
        'Hand velocity (cm/s)',
        signal,
    ])
    .astype({
        'Time from go cue (s)': float,
        'Hand velocity (cm/s)': float,
    })
    .query('`Time from go cue (s)`>=-0.3 & `Time from go cue (s)`<5')
    .reset_index(drop=True)
)

def get_pcs(df):
    pca = PCA(n_components=num_dims)
    pca.fit(np.row_stack(df[signal]))
    return pca.components_

separate_pcs = (
    td_trim
    .groupby('task')
    .apply(get_pcs)
)
_,_,vt = np.linalg.svd(np.row_stack(separate_pcs),full_matrices=False)

def project_data(df):
    sig = np.row_stack(df[signal])
    return np.dot(sig-sig.mean(axis=0),vt.T)

covar_mats = (
    td_trim
    .groupby('task')
    .apply(lambda df: pd.DataFrame(data=np.cov(project_data(df),rowvar=False)))
)

covar_mats.to_csv(f"../results/subspace_splitting/Prez_20220721_CSTRTT_{signal.replace('_rates','')}_covar_mats.csv")

# %% Check total neural variance of each task to see if we can replace separate PCA with joint PCA...


# %% import subpsace splitter data

matfile = sio.loadmat(
    f"../results/subspace_splitting/Prez_20220721_CSTRTT_{signal.replace('_rates','')}_subspacesplitter.mat",
    squeeze_me=True,
)

Q = {key: matfile['Q'][key].item() for key in matfile['Q'].dtype.names}
varexp = {key: matfile['varexp'][key].item() for key in matfile['varexp'].dtype.names}

# verify that varexp matches the covariance matrices we calculated
# np.trace(Q['unique1'].T @ covar_mats.loc['RTT'] @ Q['unique1'])/np.trace(covar_mats.loc['RTT'])

var_thresh = 0.016 # slightly arbitrary, chosen by looking at split variance explained numbers
cst_unique_proj = Q['unique1'][:,varexp['unique1_C1']>var_thresh]
rtt_unique_proj = Q['unique2'][:,varexp['unique2_C2']>var_thresh]
shared_proj = Q['shared']

# project data through the joint space into the split subspaces
td_proj = (
    td_trim.copy()
    .join(
        (
            td_trim
            .groupby('task')
            [signal]
            .apply(lambda s: s.mean(axis=0))
        ),
        on='task',
        rsuffix='_mean',
    )
    .assign(**{
        f'centered_{signal}': lambda df: df[signal] - df[f'{signal}_mean'],
        f'joint_proj_{signal}': lambda df: df.apply(lambda s: np.dot(s[f'centered_{signal}'],vt.T),axis=1),
        f'{signal}_cst_unique': lambda df: df.apply(lambda s: np.dot(s[f'joint_proj_{signal}'],cst_unique_proj),axis=1),
        f'{signal}_rtt_unique': lambda df: df.apply(lambda s: np.dot(s[f'joint_proj_{signal}'],rtt_unique_proj),axis=1),
        f'{signal}_shared': lambda df: df.apply(lambda s: np.dot(s[f'joint_proj_{signal}'],shared_proj),axis=1),
    })
    .drop(columns=[signal,f'{signal}_mean',f'centered_{signal}',f'joint_proj_{signal}'])

    .set_index(['trial_id','Time from go cue (s)'])
)

#%% k3d plots
cst_trace_plot = k3d.plot(name='CST smoothed neural traces')
max_abs_hand_vel = np.percentile(np.abs(np.row_stack(td['hand_vel'])[:,0]),95)
# plot traces
for _,trial in td.query('task=="CST"').sample(n=10).iterrows():
    neural_trace = trial['lfads_pca']
    cst_trace_plot+=k3d.line(
        neural_trace[:,0:3].astype(np.float32),
        shader='mesh',
        width=3e-3,
        attribute=trial['hand_vel'][:,0],
        color_map=k3d.paraview_color_maps.Erdc_divHi_purpleGreen,
        color_range=[-max_abs_hand_vel,max_abs_hand_vel],
    )
cst_trace_plot.display()

rtt_trace_plot = k3d.plot(name='RTT smoothed neural traces')
for _,trial in td.query('task=="RTT"').sample(n=10).iterrows():
    neural_trace = trial['lfads_pca']
    rtt_trace_plot+=k3d.line(
        neural_trace[:,0:3].astype(np.float32),
        shader='mesh',
        width=3e-3,
        attribute=trial['hand_vel'][:,0],
        color_map=k3d.paraview_color_maps.Erdc_divHi_purpleGreen,
        color_range=[-max_abs_hand_vel,max_abs_hand_vel],
    )
rtt_trace_plot.display()

#%% k3d plots with explode-y td
max_abs_hand_vel = np.percentile(np.abs(td_proj['Hand velocity (cm/s)']),95)

def plot_k3d_trace(trial,plot):
    neural_trace = np.row_stack(trial['lfads_rates_rtt_unique'])
    plot+=k3d.line(
        neural_trace[:,0:3].astype(np.float32),
        shader='mesh',
        width=3e-3,
        attribute=trial['Hand velocity (cm/s)'],
        color_map=k3d.paraview_color_maps.Erdc_divHi_purpleGreen,
        color_range=[-max_abs_hand_vel,max_abs_hand_vel],
    )
    plot.display()

# plot traces
cst_trial = td_proj.loc[227]
rtt_trial = td_proj.loc[228]
cst_trace_plot = k3d.plot(name='CST neural traces in shared space')
rtt_trace_plot = k3d.plot(name='RTT neural traces in shared space')
plot_k3d_trace(cst_trial,cst_trace_plot)
plot_k3d_trace(rtt_trial,rtt_trace_plot)


# %% add derivatives to td_proj rates

def compute_signal_derivative(td_explode,signal):
    return (
        td_explode
        .groupby('trial_id')
        [signal]
        .transform(lambda s: list(np.gradient(
            np.row_stack(s),
            s.reset_index(level=1)['Time from go cue (s)'],
            axis=0
        )))
    )

td_proj = (
    td_proj
    .assign(**{
        'lfads_rates_cst_unique_d1': lambda df: compute_signal_derivative(df,'lfads_rates_cst_unique'),
        'lfads_rates_rtt_unique_d1': lambda df: compute_signal_derivative(df,'lfads_rates_rtt_unique'),
        'lfads_rates_shared_d1': lambda df: compute_signal_derivative(df,'lfads_rates_shared'),
    })
)

# %%
# A couple things to do:
#   - Check how aligned decoder axis is with each subspace
#   - Transfer these subspaces back to original data structure to plot hand position and target info

#%% Transfer subspace rates back to original data struct (td)

td_subspace_split = (
    td
    .pipe(pyaldata.restrict_to_interval,warn_per_trial=True,epoch_fun=src.util.generate_realtime_epoch_fun(
        start_point_name='idx_goCueTime',
        rel_start_time=-0.3,
        rel_end_time=5,
    ))
    .join(
        (
            td_proj
            .groupby('trial_id')
            .agg({
                'lfads_rates_cst_unique': np.row_stack,
                'lfads_rates_rtt_unique': np.row_stack,
                'lfads_rates_shared': np.row_stack,
            })
        ),
        on='trial_id',
    )
)

# %% Plot individual traces
def plot_trial_split_space(trial_to_plot,ax_list):
    src.plot.plot_hand_trace(trial_to_plot,ax=ax_list[0],timesig='Time from go cue (s)')
    src.plot.plot_hand_velocity(trial_to_plot,ax_list[1],timesig='Time from go cue (s)')

    sig_list = ['lfads_rates_shared','lfads_rates_cst_unique','lfads_rates_rtt_unique']
    sig_colors = {
        'lfads_rates_cst_unique':'C0',
        'lfads_rates_rtt_unique':'C1',
        'lfads_rates_shared': 'C4',
    }

    rownum = 2
    for sig in sig_list:
        for dim in range(trial[sig].shape[1]):
            ax = ax_list[rownum]
            ax.plot(trial_to_plot['Time from go cue (s)'][[0,-1]],[0,0],color='k')
            ax.plot(trial_to_plot['Time from go cue (s)'],trial_to_plot[sig][:,dim],color=sig_colors[sig])
            # ax.set_yticks([])
            ax.plot([0,0],ax.get_ylim(),color='k',linestyle='--')
            sns.despine(ax=ax,trim=True)
            rownum+=1

    ax_list[-1].set_xlabel('Time from go cue (s)')

trials_to_plot = td_subspace_split.groupby('task').sample(n=1).set_index('trial_id')
fig,axs = plt.subplots(19,len(trials_to_plot),sharex=True,sharey='row',figsize=(10,18))
fig.tight_layout()
for colnum,(trial_id,trial) in enumerate(trials_to_plot.iterrows()):
    plot_trial_split_space(trial,axs[:,colnum])
# %%
