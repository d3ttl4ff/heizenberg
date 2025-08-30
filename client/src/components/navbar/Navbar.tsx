'use client';

import Image from 'next/image';

const logo = '/images/HELOGO6.png';

const Navbar = ({ scrollRef }: { scrollRef: React.RefObject<any> }) => {
  return (
    <div className="md:max-w-2xl sm:max-w-sm mx-auto w-fit  top-0 left-0 right-0 my-5 z-[50]">
      <div className="flex justify-center items-center bg-white/5 backdrop-blur-2xl text-center rounded-xl p-1">
        <div className="flex items-center justify-center gap-1 bg-white/5 rounded-lg p-1">
          <div className="rounded-md px-4 py-2 relative h-9 w-9 overflow-hidden border border-white/5 bg-accent-nvidia-dim/5">
            <Image src={logo} alt="Logo" fill className="object-contain" />
          </div>

          <div className="rounded-md px-4 py-2 text-sm text-white/60 border border-white/5">
            What is this?
          </div>

          <div
            className="rounded-md px-4 py-2 bg-accent-nvidia text-sm font-bold text-black/80 border border-white/5 cursor-pointer hover:bg-accent-nvidia-dim transition-all active:scale-95"
            onClick={() => {
              scrollRef.current?.scrollTo('#predict', {
                offset: -50,
                duration: 1,
              });
            }}
          >
            Predict Now
          </div>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
