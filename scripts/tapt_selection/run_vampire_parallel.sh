parallel --ungroup \
        python -m scripts.run_vampire \
        model_logs/vampire-sci-tokens/model.tar.gz {1} \
        --batch 64 \
        --include-package vampire \
        --predictor vampire \
        --output-file ../scientific-domains/swabha/{/.}.out \
        --silent ::: ../scientific-domains/rct_shards/*