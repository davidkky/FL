from system.flcore.servers.serverfedcf import FedCF
from system.flcore.trainmodel.models import *

def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if "mnist" in args.dataset:
            args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)

        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)
        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)
        elif args.algorithm == "FedCF":
            server = FedCF(args, i)
        else:
            raise NotImplementedError

        server.train()
        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)
    print("All done!")
    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    
    parser.add_argument('-v', "--v_smooth", type=float, default=0.1, 
                        help="Smoothing parameter for DP-noise Filter in DP-FedCF")
    
    parser.add_argument('-dp', "--privacy", type=bool, default=False, help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)

    args = parser.parse_args()

    print("=" * 50)
    print("Algorithm: {}".format(args.algorithm))
    if args.algorithm == "FedCF":
        print("DP-FedCF Smoothing (v): {}".format(args.v_smooth))
        
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("=" * 50)

    run(args)