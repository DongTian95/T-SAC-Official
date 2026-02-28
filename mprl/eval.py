from mprl.mp_exp import evaluation

if __name__ == "__main__":
    # Entire
    model_str = "artifact = run.use_artifact('dt_team/metaworld_tsac/model:version', type='model')"

    #================================================
    version_number = [1286]
    epoch = 10000

    evaluation(model_str, version_number, epoch, False)