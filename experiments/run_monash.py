import os
import pickle
from data.monash import get_datasets
from data.serialize import SerializerSettings
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from models.utils import grid_iter
from models.llmtime import get_llmtime_predictions_data
import numpy as np
# openai.api_key = os.environ['OPENAI_API_KEY']
# openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

# Specify the hyperparameter grid for each model
gpt3_hypers = dict(
    temp=0.7,
    alpha=0.9,
    beta=0,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True),
)

# 논문 저자들이 추천하는 Default 설정값
llama_hypers = dict(
    temp=0.7,        # 논문에서는 보통 0.7~1.0 사이 사용 (Llama-3는 0.7 추천)
    alpha=0.99,      # [중요] 논문 권장값 
    beta=0.3,        # [중요] 논문 권장값 (기존 0.3)
    basic=False,     # [중요] 데이터가 0 대칭(Sine파 등)이 아니면 False
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True),
)

model_hypers = {
    "ollama/forecast": {
        "model": "ollama/forecast", 
        **llama_hypers
    }
}
#model_predict_fns = {
#    'gpt-4o-mini': get_llmtime_predictions_data,
#}

def is_gpt(model):
    return ("gpt" in model) or ("o1" in model) or ("o3" in model)

# Specify the output directory for saving results
output_dir = 'outputs/monash_mini'
os.makedirs(output_dir, exist_ok=True)

models_to_run = [
   "ollama/forecast"
]

#datasets_to_run =  [
#    "weather", "covid_deaths", "solar_weekly", "tourism_monthly", "australian_electricity_demand", "pedestrian_counts",
#    "traffic_hourly", "hospital", "fred_md", "tourism_yearly", "tourism_quarterly", "us_births",
#    "nn5_weekly", "traffic_weekly", "saugeenday", "cif_2016", "bitcoin", "sunspot", "nn5_daily"
#]

datasets_to_run = [ "weather"]

#max_history_len = 50
datasets = get_datasets()
for dsname in datasets_to_run:
    print(f"Starting {dsname}")
    data = datasets[dsname]
    train, test = data
    train = data[0]
    test  = data[1]
    test= test[:1]
    train = train[:1]
    
    #train = [x[-max_history_len:] for x in train]
    test  = [x[:12] for x in test]
    # API 를 위해 적은 코드라 다시 주석처리
    #test  = [x[-max_history_len:] for x in test]
    
    if os.path.exists(f'{output_dir}/{dsname}.pkl'):
        with open(f'{output_dir}/{dsname}.pkl','rb') as f:
            out_dict = pickle.load(f)
    else:
        out_dict = {}
    
    for model in models_to_run:
        print(f"--- Debug: Steps requested (Test length): {len(test[0])} ---")
        if model in out_dict:
            print(f"Skipping {dsname} {model}")
            continue
        else:
            print(f"Starting {dsname} {model}")
        parallel = True if is_gpt(model) else False
        num_samples = 1
        
        try:
            # hypers는 grid_iter로 만들지 말고 1개만 쓰는 걸 추천 (비용/시간 폭발 방지)
            h = model_hypers[model]         
            preds = get_llmtime_predictions_data(
                train, test,
                model=h["model"],
                settings=h["settings"],
                num_samples=num_samples,
                temp=h["temp"],
                alpha=h["alpha"],
                beta=h["beta"],
                basic=h["basic"],
                parallel= False,
            )
            # print(f"--- Debug: Steps received (Preds length): {len(preds['median'][0])} ---")
            try:
                print(f"--- Debug: Steps received: {preds['median'].shape}")
            except:
                pass
            
            #preds = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=parallel)
            #medians = preds['median']
            #targets = np.array(test)
            #maes = np.mean(np.abs(medians - targets), axis=1) # (num_series)        
            medians = np.array(preds['median'])
            targets = np.array(test)
            
            # 차원 맞추기 (targets이 (1, 24)이고 medians가 (1, 24)인지 확인)
            if medians.ndim == 1: medians = medians.reshape(1, -1)
            if targets.ndim == 1: targets = targets.reshape(1, -1)

            maes = np.mean(np.abs(medians - targets), axis=1)
            
            preds['maes'] = maes
            preds['mae'] = np.mean(maes)
            out_dict[model] = preds
            
        except Exception as e:
            print(f"\n Failed {dsname} {model}")
            # 단순히 e만 찍지 말고, 상세 에러 추적(Traceback)을 출력합니다.
            import traceback
            traceback.print_exc() 
            
            # 어떤 값이 들어갔는지도 찍어보면 좋습니다.
            print(f"DEBUG - model: {h['model']}, num_samples: {num_samples}")
            continue
            
        with open(f'{output_dir}/{dsname}.pkl','wb') as f:
            pickle.dump(out_dict,f)
    print(f"Finished {dsname}")
    