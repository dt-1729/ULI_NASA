class uav_flpo:
    def __init__(self,drones,num_stations,blocks=None,
                 ugv_factor=0.0,fcr=25,distance='euclidean') -> None:
        super().__init__()
        self.den_drones = drones
        self.drones,lb_ub = normalize_drones(drones)
        self.lb,self.ub = lb_ub
        self.scale = self.ub-self.lb
        self.stations=np.repeat([0.5,0.5],num_stations,axis=0)
        self.blocks=normalize_blocks(blocks,lb_ub)
        self.N_drones= len(drones)
        self.N_stations=num_stations
        self.stage_horizon=self.N_stations+1
        self.gamma_k_length=self.N_stations+1
        self.fcr=fcr/self.scale
        self.distance=distance
        self.ugv_factor=ugv_factor
        self.bounds = [(0, 1)]*self.N_stations*2
        self.cost_normalizer = 1/(self.N_drones*(self.N_stations**self.N_stations))
        return

    def return_stagewise_cost(self,params,beta): #params is like stations
        d_F=cdist(params,params,self.distance)
        if not self.blocks==None:
            for block in self.blocks:
                d_F=d_F+cdist(params,params,metric=add_block_dist(block[0],block[1]))
        d_F=d_F+diag([my_inf]*self.N_stations)
        d_delta_to_f=array([my_inf]*self.N_stations).reshape(1,-1)
        d_df=np.concatenate((d_F,d_delta_to_f),axis=0)
        D_ss=[0]*self.N_drones
        for drone_id,drone in enumerate(self.drones):
            stage=concatenate((params,array(self.drones[drone_id][1]).reshape(1,-1)),axis=0)
            D_s=[0]*(self.stage_horizon+1)
            stage_0=array(self.drones[drone_id][0]).reshape(1,-1)
            D_s[0]=cdist(stage_0,stage,self.distance)
            if not self.blocks==None:
                for block in self.blocks:
                    D_s[0]=D_s[0]+cdist(stage_0,stage,metric=add_block_dist(block[0],block[1]))
            #print(D_s[0]-self.drones[drone_id][2]*self.fcr)
            D_s[0]=D_s[0]+penalty(D_s[0]-self.drones[drone_id][2]*self.fcr)
            #D_s[0]=D_s[0]+my_exp(beta*(D_s[0]-self.drones[drone_id][2]*self.fcr))
            #D_s[0][0,-1]=D_s[0][0,-1]*(D_s[0][0,-1]>my_inf)

            delta_id= self.N_stations+drone_id
            # so far we have taken care of the first distance matrix

            d_f_to_delta=cdist(params,array(self.drones[drone_id][1]).reshape(1,-1),self.distance)
            if not self.blocks==None:
                for block in self.blocks:
                    d_f_to_delta=d_f_to_delta+cdist(params,array(self.drones[drone_id][1]).reshape(1,-1),metric=add_block_dist(block[0],block[1]))
            d_last=np.concatenate((d_f_to_delta,array([0]).reshape(1,-1)),axis=0)
            d=np.concatenate((d_df,d_last),axis=1)


            d=d+(penalty(d-self.fcr))
            #d=d+my_exp(beta*(d-self.fcr))
            D_s[1:self.stage_horizon] = [d] * (self.stage_horizon - 1)
            d_l=[my_inf]*(self.gamma_k_length-1)
            d_l.append(0.0)
            D_s[-1]=array(d_l).reshape(-1,1)
            D_ss[drone_id]=D_s
        self.D_ss=D_ss
        return
    def calc_associations(self,beta):
        p=[]
        self.return_stagewise_cost(self.params.reshape(-1,2),beta)
        D_ss=self.D_ss
        for D_s in D_ss:
            K=len(D_s)
            D=D_s[::-1]
            out_D=[0]*(K+1)
            out_D[0]=array([0.0]).reshape(-1,1)
            out_p=[0]*(K+1)
            out_p[0]=array([1.0]).reshape(-1,1)
            out=[0]*(K+1)
            out[0]=array([1.0]).reshape(-1,1)
            for i in range(1,K+1):
                out_D[i]=(D[i-1]+repeat(transpose(out_D[i-1]),D[i-1].shape[0],axis=0))
                m=out_D[i].min(axis=1,keepdims=True)
                exp_D=exp(multiply(-beta,out_D[i]-m))
                out[i]=sum(multiply(exp_D,tile(out[i-1], (1,D[i-1].shape[0])).T),axis=1,keepdims=True)
                out_p[i]=divide(multiply(exp_D,out[i-1].T),out[i])
                out_D[i]=m
            p.append(out_p[::-1][:-1])
        self.P_ss=p
        return
    def free_energy(self,D_s,P_s,beta):
        '''
        input: D_s: a list of K numpy arrays corrosponding to distances between stages
        P_s: a list of K numpy arrays corrosponding to probabilities between stages

        output: out_c: K+1 numpy arrays with shape[1]=1, indicating the total cost of nodes
        '''

        K=len(D_s)
        D=D_s[::-1]
        P=P_s[::-1]
        out_P=[0]*(K+1)
        out_C=[0]*(K+1)
        out_H=[0]*(K+1)
        out_P[0]=array([1.0]).reshape(-1,1)
        out_C[0]=array([0.0]).reshape(-1,1)
        out_H[0]=array([0.0]).reshape(-1,1)
        for i in range(1,K+1):
          # assigning P of each node for calculating C in the next i
          out_P[i]=(P[i-1]*repeat(transpose(out_P[i-1]),P[i-1].shape[0],axis=0)).sum(axis=1).reshape(-1,1)
          out_C[i]=(P[i-1]*(D[i-1]*repeat(transpose(out_P[i-1]),D[i-1].shape[0],axis=0)+repeat(transpose(out_C[i-1]),D[i-1].shape[0],axis=0))).sum(axis=1).reshape(-1,1)
          out_H[i]=-(P[i-1]*(my_log(P[i-1])*repeat(transpose(out_P[i-1]),D[i-1].shape[0],axis=0)-repeat(transpose(out_H[i-1]),D[i-1].shape[0],axis=0))).sum(axis=1).reshape(-1,1)
        # D-1/beta*H
        return (out_C[-1].T).sum() + (-1/beta)*(out_H[-1].T).sum()
    def free_energy_Gibbs(self,D_s,beta):
        K=len(D_s)
        D=D_s[::-1]
        out_D=[0]*(K+1)
        out_D[0]=array([0.0]).reshape(-1,1)
        out=[0]*(K+1)
        out[0]=array([1.0]).reshape(-1,1)
        for i in range(1,K+1):

            out_D[i]=(D[i-1]+repeat(transpose(out_D[i-1]),D[i-1].shape[0],axis=0))

            m=out_D[i].min(axis=1,keepdims=True)
            exp_D=exp(multiply(-beta,D[i-1]))
            out[i]=sum(multiply(exp_D,tile(out[i-1], (1,D[i-1].shape[0])).T),axis=1,keepdims=True)
            out_D[i]=m
        if isclose(out[-1],0.0).all():
            return m.sum()
        else:
            return (-1/beta*log(out[-1]).sum())

    def objective(self,params,beta):
        self.return_stagewise_cost(params.reshape(-1,2),beta)
        cost=0
        for i in range(len(self.D_ss)):
            cost+=self.free_energy_Gibbs(self.D_ss[i],beta)
        if self.ugv_factor == 0.0:
            return self.cost_normalizer*cost
        else:
            return self.cost_normalizer*cost+self.ugv_factor*linalg.norm(params.reshape(-1,2)-self.stations)

    def optimize_D(self,init_guess,beta,method):
        # bounds=(np.min([np.min(drone[0]+drone[1]) for drone in drones]),np.max([np.max(drone[0]+drone[1]) for drone in drones]))*len(init_guess)
        result = minimize(self.objective, init_guess,args=(beta,),bounds=self.bounds,method=method)
        self.params = result.x
        self.cost_fun=result.fun
    def calc_routs(self):
        O=[]
        for i in range(self.N_drones):
          m=0
          o=[]
          for p in self.P_ss[i]:
              m=argmax(p[m,:])
              o.append(m)
          o.pop()
          O.append(o)
        self.routs=O

    def train(self,beta_init=1e-6,beta_f=100,alpha=1.5,purturb=0.1,method='powell',verbos=0):
        self.Y_s=[]
        self.Betas=[]
        self.params=ndarray.flatten(self.stations)
        beta=beta_init
        # self.return_stagewise_cost(self.params.reshape(-1,2))
        old_cost=my_inf
        while beta <= beta_f:
            count=0
            self.params=self.params+np.random.normal(0, purturb, self.params.shape)

            self.optimize_D(self.params,beta,method=method) #based on P_ss
            count+=1
            if verbos:
              print(f'Beta: {beta:.4e}  Cost: {self.cost_fun:0.5e}')
            # if abs(self.cost_fun-old_cost) <= 1e-6:
            #     print("--Optimization Terminated--")
            #     break
            old_cost=self.cost_fun
            beta=beta*alpha
            self.Y_s.append(self.params.reshape(-1,2))
            self.Betas.append(beta)
        self.calc_associations(beta)
        self.calc_routs()

    def print_routs(self):
        print("")
        for i,o in enumerate(self.routs):
          print(f'\nDrone {i+1} --->', end='')
          for j in o:
            if j<env.N_stations:
              print(f'f{j+1} --->', end='')
            else:
              print(f'[D{i+1}]', end='')
              break
    def return_total_cost(self):
        return self.cost_fun*self.scale/self.cost_normalizer
    def return_direct_cost(self):
        return np.sum([np.sum((np.array(drone[0])-np.array(drone[1]))**2)**0.5 for drone in self.den_drones])
    def plot_routs(self,show_info=True,show_nums=True,save=False,show_ugv=False):
        state_locs=self.params.reshape(-1,2).copy()*self.scale+self.lb
        den_drones = self.den_drones
        drone_locs=array([i[0] for i in den_drones])
        dest_locs=array([i[1] for i in den_drones])
        plt.scatter(drone_locs[:,0],drone_locs[:,1],color='black',label='UAVs')
        for i, loc in enumerate(drone_locs):
            if show_info:
                plt.text(loc[0], loc[1], 'V'+str(i+1)+f' ({self.drones[i][2]})', ha='center', va='bottom')
        plt.scatter(state_locs[:,0],state_locs[:,1],marker='^',label='UGVs')
        f_indices = np.argsort(state_locs[:,0])
        for i, loc in enumerate(state_locs):
            if show_info:
                plt.text(loc[0], loc[1], 'F'+str(i+1), ha='center', va='bottom')
        plt.scatter(dest_locs[:,0],dest_locs[:,1],marker='*',label='Dest.')
        for i, loc in enumerate(dest_locs):
            if show_info:
                plt.text(loc[0]+1*np.random.rand()*(-1)**round(np.random.rand()), loc[1]+1*np.random.rand(), 'D'+str(i+1), ha='center', va='bottom')
        options = ['-', '--', '-.', ':',]
        modified_lines = []
        colors = []
        styles = []
        for i, o in enumerate(self.routs):
          
            line_style=np.random.choice(options)
            styles.append(line_style)
            drone_loc = drone_locs[i]
            try:
              state_loc = state_locs[o[0]]
            except:
              state_loc=dest_locs[i]
            color=np.random.rand(3)
            colors.append(color)
            plt.plot([drone_loc[0], state_loc[0]], [drone_loc[1], state_loc[1]], color=color,linewidth=1.5,linestyle=line_style)
            dist = np.sqrt(np.sum((drone_loc - state_loc) ** 2))
            col='red' if dist > self.drones[i][2]*self.fcr*self.scale else 'green'
            if dist >0.0:
              if show_nums:
                plt.text((drone_loc[0] + state_loc[0]) / 2, (drone_loc[1] + state_loc[1]) / 2, f'{dist:.2f}', color=col, ha='center')
            if len(o)>1:
              for j in range(len(o) - 1):
                  try:
                    loc1 = state_locs[o[j]]
                  except:
                    loc1 = dest_locs[i]
                  try:
                    loc2 = state_locs[o[j + 1]]
                  except:
                    loc2=dest_locs[i]
                  plt.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], color=color,linewidth=1.5,linestyle=line_style)
                  dist = np.sqrt(np.sum((loc1 - loc2) ** 2))
                  col='red' if dist > self.fcr*self.scale else 'green'
                  if dist>0.0:
                    if show_nums:
                      plt.text((loc1[0] + loc2[0]) / 2, (loc1[1] + loc2[1]) / 2, f'{dist:.2f}', color=col, ha='center')
        if not self.blocks==None:
            counter=0
            den_blocks = denormalize_blocks(self.blocks,(self.lb,self.ub))
            for center, radius in den_blocks:
                c_x, c_y = center
                label='Obst.' if counter==0 else None
                circle=patches.Circle(center, radius,edgecolor="white", linewidth=0.05,facecolor='grey', linestyle='-',label=label)
                circle.set_zorder(2)
                plt.gca().add_patch(circle)
                counter+=1
        ax = plt.gca()

        # Process each drone, but only draw the arcs this time
        counter = 0
        label=None
        drone_id=0
        for drone, route in zip(den_drones, self.routs):
            path_segments = get_drone_path_segments(drone, route, state_locs)
            for segment in path_segments:
                for block in den_blocks:
                    intersections = line_circle_intersection(block[0], block[1], segment[0], segment[1])
                    if intersections:
                        if counter==0:
                            label='Adj.'
                            counter +=1
                        theta_start, theta_end = calculate_shorter_arc(block[0], block[1], intersections[0], intersections[1])
                        arc = patches.Arc(block[0], 2*block[1], 2*block[1],
                                                     theta1=np.degrees(theta_start), theta2=np.degrees(theta_end),
                                                     color=colors[drone_id],linewidth=3, linestyle=styles[drone_id])
                        ax.add_patch(arc)
                    label=None
            drone_id +=1
        if show_ugv:
            plt.scatter(self.stations[0,0],self.stations[0,1],color='orange',marker='2')
            plt.text(self.stations[0,0], self.stations[0,1], 'Initial UGV location', ha='center', va='bottom')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title(f'M={self.N_stations} F.C.R. = {self.fcr*self.scale}')
        plt.gca().set_aspect('equal')

        if save:
          plt.savefig(f'sim res for M={self.N_stations} F.C.R. = {self.fcr*self.scale}.pdf')
        plt.show()