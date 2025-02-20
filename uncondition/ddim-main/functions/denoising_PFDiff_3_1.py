import torch


def generalized_steps(x, seq, model, alpha_dict, args, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        seq_next_next = [-2] + [-1] + list(seq[:-2])
        seq_next_next_next = [-3] + [-2] + [-1] + list(seq[:-3])
        seq_next_next_next_next = [-4] + [-3] + [-2] + [-1] + list(seq[:-4])
        x0_preds = []
        xs = [x]
        eta = kwargs.get("eta", 0)

        # --------------------------begin---------------------------------------
        et_xt_list = []

        def p_theta_any(theta, xt, t, t_next):
            at = alpha_dict[t]
            at_next = alpha_dict[t_next]
            x0_t = (xt - theta * (1 - at).sqrt()) / at.sqrt()

            c1 = (
                eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next_mu = at_next.sqrt() * x0_t  + c2 * theta
            if t_next == -1:
                c1 = 0
            x_next_sigma = c1 * torch.randn_like(xt)

            return xt_next_mu, x_next_sigma
        #----------------------------end----------------------------------------

        for index, (i, j, k, l, m) in enumerate(zip(reversed(seq), reversed(seq_next), reversed(seq_next_next), reversed(seq_next_next_next), reversed(seq_next_next_next_next))):
            if index != 0 and (index-1) % 4 !=0:
                continue
            if len(et_xt_list) == 0:
                t = (torch.ones(n) * i).to(x.device)
                xt = xs[-1].to('cuda')
                et = model(xt, t)

                xt_next_mu, x_next_sigma = p_theta_any(et, xt, i, j)
                xt_next = xt_next_mu + x_next_sigma
                
                et_xt_list.append([et, xt, i])
                xs.append(xt_next.to('cpu'))
            else:
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                next_next_t = (torch.ones(n) * k).to(x.device)
                next_next_next_t = (torch.ones(n) * l).to(x.device)
                xt = xs[-1].to('cuda')

                if l == -1:
                    et_, xt_, t_ = et_xt_list[-1]
                    xt_next_mu, x_next_sigma = p_theta_any(et_, xt, i, j)
                    xt_next = xt_next_mu + x_next_sigma
                    et_next = model(xt_next, next_t)

                    xt_next_next_next_mu, x_next_next_next_sigma = p_theta_any(et_next, xt, i, l)
                    xt_next_next_next = xt_next_next_next_mu + x_next_next_next_sigma
                    et_xt_list.append([et_next, xt_next, j])
                    xs.append(xt_next_next_next.to('cpu'))
                    break


                if k == -1:
                    et_, xt_, t_ = et_xt_list[-1]
                    xt_next_mu, x_next_sigma = p_theta_any(et_, xt, i, j)
                    x_next = xt_next_mu + x_next_sigma
                    et_next = model(x_next, next_t)
                    xt_next_next_mu, x_next_next_sigma = p_theta_any(et_next, xt, i, k)
                    xt_next_next = xt_next_next_mu + x_next_next_sigma
                    et_xt_list.append([et_next, x_next, j])
                    xs.append(xt_next_next.to('cpu'))
                    break

                if j == -1:
                    et = model(xt, t)
                    xt_next_mu, x_next_sigma = p_theta_any(et, xt, i, j)
                    xt_next = xt_next_mu + x_next_sigma
                    
                    xs.append(xt_next.to('cpu'))
                    break

                # --------------------------begin---------------------------------------
                et_, xt_, t_ = et_xt_list[-1]
                xt_next_mu, x_next_sigma = p_theta_any(et_, xt, i, j)
                x_next = xt_next_mu + x_next_sigma
                et_next = model(x_next, next_t)

                xt_next_next_next_next_mu, x_next_next_next_next_sigma = p_theta_any(et_next, xt, i, m)
                xt_next_next_next_next = xt_next_next_next_next_mu + x_next_next_next_next_sigma
                et_xt_list.append([et_next, x_next, j])
                xs.append(xt_next_next_next_next.to('cpu'))

                if m == -1:
                    break

                #----------------------------end----------------------------------------

    return xs, x0_preds


def ddpm_steps(x, seq, model, alpha_dict, args, **kwargs):
    pass
