from metric import FDFR, ISM, SER_FIQ, BRISQUE
import logging

def parse_args(input_args = None):
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument( # --target_dir )
        '--target_dir',
        type = str,
        default = None,
        required = True,
        help = "Target folder that will be evaluated"
    )

    parser.add_argument( # --identity_dir )
        '--identity_dir',
        type = str,
        default = None,
        required = False,
        help = "Identity folder that will be used to calculate cosine similarity in ISM metric."
    )

    parser.add_argument( # --log_into='logs/eval/log.log' )
        '--log_into',
        type = str,
        default = 'logs/eval/log.log',
        required = None,
        help = "Path to log file"
    )

    parser.add_argument( # --metric='all' )
        '--metric',
        type = str,
        default = 'all',
        required = None,
        help = "Type of metric to evaluate. 'all' (default), 'fdfr', 'ism', 'ser-fiq', 'brisque'"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def get_logger(name, type, log_into):
    if type == 'general':
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_into)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    elif type == 'event':
        logger = logging.getLogger('special_logger')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_into)
        formatter = logging.Formatter('[%(asctime)s] - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def get_path(parameterized_path, parameters):
    """
        Return a true path from a parameterized path, given the parameters
    """
    parts = parameterized_path.split('/')

    for key, value in parameters:
        for i, part in enumerate(parts):
            if part == key:
                parts[i] = str(value)

    return '/'.join(parts)

def main(args):
    target_dir = args.target_dir
    identity_dir = args.identity_dir
    metric = args.metric
    log_into = args.log_into
    
    event_logger = get_logger(
        name = 'eval_event_logger',
        type = 'event',
        log_into = log_into
    )

    general_logger = get_logger(
        name = 'eval_general_logger',
        type = 'general',
        log_into = log_into
    )
    
    fdfr = None
    ism = None
    ser_fiq = None
    brisque = None
    
    if metric == 'all':
        if identity_dir == None:
            raise RuntimeError("eval.py - main(): If metric == 'all' or 'ism', must specify identity_dir")
        
        fdfr = FDFR.eval(target_dir, log_info = True)
        ism = ISM.eval(target_dir, identity_dir, log_info = True)
        ser_fiq = SER_FIQ.eval(target_dir)
        brisque = BRISQUE.eval(target_dir)
    elif metric == 'fdfr':
        fdfr = FDFR.eval(target_dir)
    elif metric == 'ism':
        if identity_dir == None:
            raise RuntimeError("eval.py - main(): If metric == 'all' or 'ism', must specify identity_dir")
        
        ism = ISM.eval(target_dir, identity_dir)
    elif metric == 'ser-fiq':
        ser_fiq = ser_fiq = SER_FIQ.eval(target_dir)
    elif metric == 'brisque':
        brisque = BRISQUE.eval(target_dir)

    general_logger.info(f"")
    event_logger.info(f"eval.py - main():")
    general_logger.info(f"    metric = {metric}")
    general_logger.info(f"    target_dir = {target_dir}")
    general_logger.info(f"    identity_dir = {identity_dir}")
    general_logger.info(f"--------------------")
    if fdfr:
        general_logger.info(f"FDFR = {fdfr:.2f}")
    if ism:
        general_logger.info(f"ISM = {ism:.2f}")
    if ser_fiq:
        general_logger.info(f"SER-FQI = {ser_fiq}")
    if brisque:
        general_logger.info(f"BRISQUE = {brisque}")
    general_logger.info(f"")

if __name__ == '__main__':
    args = parse_args()
    main(args) 
