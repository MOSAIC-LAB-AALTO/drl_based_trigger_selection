
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parsing the type of DRL/RL to be tested')
    parser.add_argument('-t','--train', help='Train DRL/RL')
    parser.add_argument('-o', '--observe', help='Observe a trained DRL/RL', required=True)
    args = vars(parser.parse_args())
    print(args)
    time = 100
    if time >= 99 and (args['train'] != "dqn" and args['train'] != "dqn_batch"):
        print("update")
    if args['train'] is not None:
        print('WOWO')