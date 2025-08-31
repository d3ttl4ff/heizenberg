import React from 'react';
import { AtomicTitle } from '@/components/ui';

function About() {
  return (
    <div className="md:max-w-4xl sm:max-w-sm sm:mx-auto top-0 left-0 right-0 my-48 mx-4">
      <div className="flex flex-col items-center">
        <AtomicTitle title="About" />

        <div className="flex flex-col gap-8">
          <div className="text-center text-xl">
            <span className="font-bold text-accent-nvidia">Heizenberg</span> is
            a predictive analytics system designed for the PC gaming industry.
            In today’s oversaturated Steam market, developers face high risks
            bringing new titles to life. Heizenberg uses a large-scale custom
            dataset and advanced
            <span className="font-bold text-cyan-500"> Machine Learning </span>
            models to forecast key success metrics such as owners, player
            counts, copies sold, and revenue - all before a game is released.
          </div>

          <div className="text-center text-xl">
            To make predictions transparent and trustworthy, Heizenberg
            integrates
            <span className="font-bold text-cyan-500">
              {' '}
              Explainable AI (XAI)
            </span>
            . This allows developers to see not only the forecasts but also the
            reasons behind them, highlighting which features drive predicted
            outcomes. Beyond forecasting, the system offers actionable
            post-prediction insights — guiding strategies for pricing, platform
            choice, and market positioning. By combining accuracy with
            interpretability, Heizenberg empowers indie and mid-scale developers
            to minimize financial risk and align creative visions with market
            demands.
          </div>
        </div>
      </div>
    </div>
  );
}

export default About;
