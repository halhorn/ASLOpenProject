from argparse import ArgumentParser
import json
import os
import wgan_estimator

def run():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--noise_dims", type=int, default=64)
    parser.add_argument("--generator_lr", type=float, default=0.000076421)
    parser.add_argument("--discriminator_lr", type=float, default=0.0031938)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_train_steps", type=int, default=20000)
    parser.add_argument("--num_eval_steps", type=int, default=100)

    parser.add_argument("--job-dir", default='junk')
    
    params = {k: v for k, v in parser.parse_args().__dict__.items() if v is not None}
    print("hyper parameters are {}".format(params))
    
    output_dir = os.path.join(
        params['model_dir'],
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )
    
    wgan_estimator.train_and_evaluate(params)
    

if __name__ == '__main__':
    run()
