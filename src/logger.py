import logging 
import os 
import datetime 


LOG_File=f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

loger_path=os.path.join(os.getcwd(),"TitanicData_ml_project_\logs", LOG_File)
os.mkdir(loger_path)

LOG_fILE_PATH= os.path.join(loger_path,LOG_File)

logging.basicConfig(
    filename=LOG_fILE_PATH,
    format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    level=logging.INFO

)
if __name__ == '__main__':
    logging.info('Logging file Starting')

