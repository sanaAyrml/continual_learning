from utils_incremental.compute_features import compute_features
from utils_incremental.compute_accuracy import compute_accuracy
from utils_incremental.compute_confusion_matrix import compute_confusion_matrix
from utils_incremental.incremental_train_and_eval import incremental_train_and_eval
from utils_incremental.incremental_train_and_eval_MS import incremental_train_and_eval_MS
from utils_incremental.incremental_train_and_eval_LF import incremental_train_and_eval_LF
from utils_incremental.incremental_train_and_eval_MR_LF import incremental_train_and_eval_MR_LF
from utils_incremental.incremental_train_and_eval_AMR_LF import incremental_train_and_eval_AMR_LF
from utils_incremental.incremental_train_and_eval_MR_LF_CL import incremental_train_and_eval_MR_LF_CL

           
def train_model(trainloader, testloader,model,ref_model,ckp_name,main_ckp_prefix,optimizer,lr_scheduler,args,iteration_total,iteration,start_iter,cur_lamda,device,mode = "verifier",train_mode = "no_sampler",evalloader =None):   
    
    
    ###############################
    model = model.to(device)
    #############################
    log_f_name_train = './sweep_checkpoint/{}_train_log_{}_run_{}_iteration_{}.csv'.format(main_ckp_prefix, mode,iteration_total, iteration)
    print(log_f_name_train)

    log_f_name_test = './sweep_checkpoint/{}_test_log_{}_run_{}_iteration_{}.csv'.format(main_ckp_prefix, mode,iteration_total, iteration)
    print(log_f_name_test)
    if args.less_forget and args.mr_loss  and args.CL > 0:
        print(mode+" incremental_train_and_eval_MR_LF_CL")
        # if iteration > start_iter:
        model = incremental_train_and_eval_MR_LF_CL(ckp_name,\
                                                 log_f_name_train,\
                                                 log_f_name_test,\
                                                 args.epochs, \
                                                 model, ref_model, \
                                                 optimizer, \
                                                 lr_scheduler, \
                                                 trainloader, testloader, evalloader,\
                                                 iteration, start_iter, \
                                                 cur_lamda, \
                                                 args.dist, args.K, args.lw_mr, args.CL,args.ro ,device=device,alpha_3= args.alpha_3) 
    elif args.less_forget and args.mr_loss :
        print(mode+" incremental_train_and_eval_MR_LF")
        # if iteration > start_iter:
        model = incremental_train_and_eval_MR_LF(ckp_name,\
                                                 log_f_name_train,\
                                                 log_f_name_test,\
                                                 args.epochs, \
                                                 model, ref_model, \
                                                 optimizer, \
                                                 lr_scheduler, \
                                                 trainloader, testloader, \
                                                 iteration, start_iter, \
                                                 cur_lamda, \
                                                 args.dist, args.K, args.lw_mr, device=device,alpha_3= args.alpha_3) 
        
    elif args.less_forget and args.amr_loss:
        print(mode+" incremental_train_and_eval_AMR_LF")
        tg_model = incremental_train_and_eval_AMR_LF(args.epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, \
            iteration, start_iter, \
            cur_lamda, \
            args.dist, args.K, args.lw_mr)
    else:
        if args.less_forget:
            print(mode+" incremental_train_and_eval_LF")
            tg_model = incremental_train_and_eval_LF(args.epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
                trainloader, testloader, \
                iteration, start_iter, \
                cur_lamda)
        else:
            if args.mimic_score:
                print(mode+" incremental_train_and_eval_MS")
                tg_model = incremental_train_and_eval_MS(args.epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
                    trainloader, testloader, \
                    iteration, start_iter,
                    args.lw_ms)
            else:                         
                print(mode+" incremental_train_and_eval")
                tg_model = incremental_train_and_eval(args.epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
                    trainloader, testloader, \
                    iteration, start_iter,
                    args.T, args.beta) 
    return model