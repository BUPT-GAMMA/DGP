import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--DS_pair', default=None)    
    parser.add_argument('--DS_ood', default=None, help='Out-of-distribution dataset')
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)

    parser.add_argument('--lr', dest='lr', type=float, default=0.01,
            help='Learning rate: 0.1, 0.01, 0.001')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
            help='Hidden dimension of GIN')
    parser.add_argument('--proj-hidden-dim', dest='proj_hidden_dim', type=int, default=96,
            help='Hidden dimension of projection head')
    parser.add_argument('--proj-output-dim', dest='proj_output_dim', type=int, default=96,
            help='Output dimension of projection head')

    parser.add_argument('--aug', type=str, default='none', 
            choices=['none', 'dnodes', 'pedges', 'mask_nodes', 'subgraph', 'random2', 'random3', 'random4'],
            help='GCL augmentation type')
    parser.add_argument('--eta', type=float, default=1.0,
            choices=[0.1, 1.0, 10.0, 100.0, 1000.0],
            help='SimGRACE perturbation weight')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
           
    # DGP parameters
    parser.add_argument('--lambda_', type=float, default=0.1,
            choices=[0.1, 0.5, 1.0, 5.0, 10.0],
            help='Decoupling loss weight')
    parser.add_argument('--alpha_1', type=float, default=100,
            choices=[100, 1000, 10000, 100000],
            help='Class-specific distance loss weight')
    parser.add_argument('--alpha_2', type=float, default=100,
            choices=[100, 1000, 10000, 100000],
            help='Class-agnostic distance loss weight')
    parser.add_argument('--gamma', type=float, default=0.1,
            choices=[0.1, 0.5, 1.0, 5.0, 10.0],
            help='Test phase balance parameter')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--dgp_lr', type=float, default=0.001,
            help='DGP learning rate in range [1e-4, 1e-1]')
            
    # Model type
    parser.add_argument('--model_type', type=str, default='gcl',
            choices=['gcl', 'gcl-ft', 'dgp-gcl', 'simgrace', 'simgrace-ft', 'dgp-sim'],
            help='Model type to use')    
    return parser.parse_args()

